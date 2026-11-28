"""OPENMM-1: the LAMMPS model QuantaBench writes, rebuilt in OpenMM (``md.openmm_engine``), term by term against
LAMMPS ``run 0`` on the same minimized configuration.

Each case writes a small liquid box with ``md.liquid`` (lattice packing), minimizes it in LAMMPS, and compares the
bonded energies per style and the nonbonded total (Coulomb, Lennard-Jones, 1-4 pairs, the tail where the stage has
it) with OpenMM's CPU platform: OPLS-AA (geometric mixing, ``opls`` dihedrals, ``cvff`` impropers) and a GAFF2-like
model (``fourier`` dihedrals, Lorentz-Berthelot, Coulomb 1-4 × 1/1.2), each with a plain LJ cutoff (with and without
the tail) and with full long-range dispersion (LAMMPS ``pppm/disp``, OpenMM LJ-PME). Needs an ``lmp`` QuantaBench
finds and a Python with OpenMM (``envs/openmm.yml``); skipped otherwise (CI has neither)."""

import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantabench.md import forcefield as F
from quantabench.md import liquid as L
from quantabench.md import openmm_engine as OE

C6F14 = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
C6F6 = "Fc1c(F)c(F)c(F)c(F)c1F"


@pytest.fixture(scope="module")
def lmp():
    from quantabench.md import bench

    info, why = bench.find_lammps()
    if info is None:
        pytest.skip(why)
    if not info.has("KSPACE") or not info.has("MOLECULE"):
        pytest.skip(f"LAMMPS {info.version} lacks KSPACE or MOLECULE")
    return info


@pytest.fixture(scope="module")
def omm_python():
    py, why = OE.find_openmm_python()
    if py is None:
        pytest.skip(why)
    return py


def gaff_like(tm: F.TypedMolecule) -> F.TypedMolecule:
    """The OPLS-AA molecule as GAFF2 writes its terms: each OPLS dihedral term a ``fourier`` term of its own,
    Lorentz-Berthelot mixing, Coulomb 1-4 scaled by 1/1.2 (the impropers stay ``cvff``)."""
    d = tm.to_dict()
    types, rows = {}, []
    for i, j, k, l_, name in tm.dihedrals:
        for n, kn in enumerate(tm.dihedral_types[name].coeffs[:4], start=1):
            if kn == 0.0:
                continue
            key = f"{name}:n{n}"
            types[key] = {"style": "fourier", "coeffs": [0.5 * kn, n, 0.0 if n % 2 else 180.0], "source": "test"}
            rows.append([i, j, k, l_, key])
    d.update(dihedral_types=types, dihedrals=rows, mixing="arithmetic", special_coul=[0.0, 0.0, 1 / 1.2],
             styles={**tm.styles, "dihedral": "fourier"}, forcefield="gaff2-like")
    return F.TypedMolecule.from_dict(d)


@pytest.fixture(scope="module")
def molecules():
    c6f14 = F.type_molecule(C6F14, "C6F14", "oplsaa")
    c6f6 = F.type_molecule(C6F6, "C6F6", "oplsaa")
    return {"c6f14": c6f14, "c6f6": c6f6, "c6f6-gaff": gaff_like(c6f6), "c6f14-gaff": gaff_like(c6f14)}


def _run0_input(tm, cfg, stage: str) -> str:
    lines = ["log none", *L._variables(cfg), *L._styles(tm, cfg, stage), "read_data system.data",
             "include ff.params", *L._kspace(tm, cfg), "neighbor 2.0 bin", "neigh_modify delay 0 every 1 check yes",
             "thermo_style custom step pe ebond eangle edihed eimp evdwl ecoul elong",
             "min_style cg", "minimize 1.0e-4 1.0e-6 300 3000", "reset_timestep 0",
             "write_data min.data nocoeff",
             "run 0",
             'print "QB_E $(pe:%.10f) $(ebond:%.10f) $(eangle:%.10f) $(edihed:%.10f) $(eimp:%.10f) $(evdwl:%.10f) '
             '$(ecoul:%.10f) $(elong:%.10f)"']
    return "\n".join(lines) + "\n"


def _lammps_energies(lmp, folder: Path, tm, cfg, stage: str) -> dict:
    (folder / "in.run0").write_text(_run0_input(tm, cfg, stage), encoding="utf-8")
    out = subprocess.run([lmp.path, "-nocite", "-in", "in.run0"], cwd=folder, capture_output=True, text=True,
                         timeout=600)
    line = [x for x in out.stdout.splitlines() if x.startswith("QB_E ")]
    assert out.returncode == 0 and line, out.stdout[-3000:] + out.stderr[-2000:]
    v = [float(x) for x in line[-1].split()[1:]]
    return {"pe": v[0], "bond": v[1], "angle": v[2], "dihedral": v[3], "improper": v[4], "nonbonded": sum(v[5:8]),
            "scale": sum(abs(x) for x in v[1:])}


def _openmm_energies(python: str, folder: Path, stage: str, **override) -> dict:
    meta = json.loads((folder / "liquid_setup.json").read_text(encoding="utf-8"))
    phys = replace(OE.physics_from_setup(meta, stage), **override)
    cmd = [python, "-m", "quantabench.md.openmm_engine", "energies", str(folder / "min.data"),
           str(folder / "ff.params"), "--physics", json.dumps(phys.to_dict())]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=OE.openmm_env(python, 2))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


CASES = [  # (molecule, dispersion, stage)
    ("c6f14", "cutoff12", "slab"),        # OPLS-AA, geometric table, no tail
    ("c6f14", "cutoff12", "bulk"),        # ... with the homogeneous tail
    ("c6f14", "ewald", "slab"),           # LJ-PME, geometric (LB + the geometric-minus-LB table)
    ("c6f6", "cutoff12", "slab"),         # aromatic: cvff impropers
    ("c6f6-gaff", "cutoff12", "bulk"),    # fourier dihedrals, LB natively, 1-4 Coulomb 1/1.2, tail
    ("c6f14-gaff", "ewald", "slab"),      # LB + LJ-PME (LAMMPS mix/disp geom)
]


@pytest.mark.parametrize(("mol", "dispersion", "stage"), CASES)
def test_energy_parity_with_lammps(lmp, omm_python, molecules, tmp_path, mol, dispersion, stage):
    tm = molecules[mol]
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=25.5, packing="lattice", dispersion=dispersion,
                             kspace_accuracy=1e-6, dump_every=0)
    L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.6 if "c6f14" in mol else 1.5)
    ref = _lammps_energies(lmp, tmp_path, tm, L.load_folder(tmp_path)[1], stage)
    got = _openmm_energies(omm_python, tmp_path, stage, ewald_tolerance=1e-6)
    for term in ("bond", "angle", "dihedral", "improper"):
        assert got[term] == pytest.approx(ref[term], rel=1e-6, abs=1e-4), (term, got, ref)
    tol = 2e-4 * ref["scale"] + 0.05
    assert abs(got["nonbonded"] - ref["nonbonded"]) < tol, (got, ref)
    assert abs(got["pe"] - ref["pe"]) < tol + 1e-3, (got, ref)


def test_explicit_pair_coeff_goes_through_the_table(lmp, omm_python, molecules, tmp_path):
    """An off-diagonal ``pair_coeff i j`` (a refit's cross term) overrides the mixing rule in both engines."""
    tm = molecules["c6f14"]
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=25.5, packing="lattice", dispersion="cutoff12",
                             kspace_accuracy=1e-6, dump_every=0)
    L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.6)
    with open(tmp_path / "ff.params", "a", encoding="utf-8") as fh:
        fh.write("pair_coeff 1 2 0.08 3.10  # test cross term\n")
    ref = _lammps_energies(lmp, tmp_path, tm, L.load_folder(tmp_path)[1], "slab")
    got = _openmm_energies(omm_python, tmp_path, "slab", ewald_tolerance=1e-6)
    assert abs(got["nonbonded"] - ref["nonbonded"]) < 2e-4 * ref["scale"] + 0.05, (got, ref)


# --------------------------------------------------------------------------------------- without LAMMPS or OpenMM
def test_readers_and_torsion_terms(molecules, tmp_path):
    tm = molecules["c6f6"]
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=25.5, packing="lattice", dispersion="cutoff12",
                             dump_every=0)
    s = L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.5)
    m = OE.load_model(tmp_path / "system.data", tmp_path / "ff.params")
    assert m.n_atoms == s.n_atoms and len(m.molecules()) == s.n_molecules
    assert len(m.impropers) == 6 * s.n_molecules and m.velocities is None
    assert np.allclose(m.box_len, s.box) and math.isclose(m.charges.sum(), 0.0, abs_tol=1e-6)
    phys = OE.physics_from_setup(OE.read_setup(tmp_path), "bulk")
    assert phys.mixing == "geometric" and phys.lj == "cutoff" and phys.tail and phys.cutoff_A == 12.0
    assert not OE.physics_from_setup(OE.read_setup(tmp_path), "slab").tail
    # opls: K_n / 2 with phase 0 (odd n) or pi (even n); cvff K d n: phase 0 for d = +1, pi for d = -1
    terms = OE._torsions("opls", [1.0, -2.0, 3.0, 0.0])
    assert [(n, round(p, 6)) for n, p, _k in terms] == [(1, 0.0), (2, round(math.pi, 6)), (3, 0.0)]
    assert terms[1][2] == pytest.approx(-1.0 * OE.KCAL)
    assert OE._torsions("cvff", [2.5, -1, 2]) == [(2, math.pi, 2.5 * OE.KCAL)]
    assert OE._torsions("fourier", [2, 1.0, 1, 0.0, 0.5, 3, 180.0]) == [(1, 0.0, OE.KCAL), (3, math.pi, 0.5 * OE.KCAL)]
    with pytest.raises(OE.OpenMMModelError):
        OE._torsions("cvff", [1.0, 0.5, 2])


def test_refuses_what_it_cannot_translate(molecules, tmp_path):
    pytest.importorskip("openmm")
    tm = molecules["c6f14"]
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=25.5, packing="lattice", dispersion="cutoff12",
                             dump_every=0)
    L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.6)
    m = OE.load_model(tmp_path / "system.data", tmp_path / "ff.params")
    phys = OE.physics_from_setup(OE.read_setup(tmp_path), "slab")
    with pytest.raises(OE.OpenMMModelError, match="dihedral_style charmm"):
        OE.build_system(m, replace(phys, styles={**phys.styles, "dihedral": "charmm"}))
    with pytest.raises(OE.OpenMMModelError, match="special_bonds"):
        OE.build_system(m, replace(phys, special_lj=(0.0, 0.5, 0.5)))
    with pytest.raises(OE.OpenMMModelError, match="twice the cutoff"):
        OE.build_system(m, replace(phys, cutoff_A=float(m.box_len.min()) / 2 + 0.5))


# --------------------------------------------------------------------------------------- the run's pieces (no OpenMM)
def test_dcd_append_read_truncate(tmp_path):
    from quantabench.md import dcd, sim

    rng = np.random.default_rng(1)
    frames = [rng.normal(size=(7, 3)) * 10 for _ in range(5)]
    w = dcd.DCDWriter(tmp_path / "t.dcd", 7, 10, 0.01)
    for k, x in enumerate(frames[:3]):
        w.write(k * 10, x, (11.0, 12.0, 13.0))
    w2 = dcd.DCDWriter(tmp_path / "t.dcd", 7, 10, 0.01)           # a later segment continues the file
    assert w2.next_step() == 30
    with pytest.raises(ValueError):
        w2.write(40, frames[3], (11.0, 12.0, 13.0))
    w2.write(30, frames[3], (11.0, 12.0, 13.0))
    w2.write(40, frames[4], (11.0, 12.0, 14.0))
    got = list(dcd.read_frames(tmp_path / "t.dcd"))
    assert [s for s, _x, _b in got] == [0, 10, 20, 30, 40]
    assert np.abs(got[4][1] - frames[4]).max() < 1e-5 and list(got[4][2]) == [11.0, 12.0, 14.0]
    assert [s for s, _x, _b in dcd.read_frames(tmp_path / "t.dcd", after=20)] == [30, 40]
    assert sim.cut_at(tmp_path / "t.dcd", "dcd", 20, keep_step=False) == 3      # a resume at step 20 rewrites it
    assert dcd.frame_steps(tmp_path / "t.dcd") == [0, 10]
    with open(tmp_path / "t.dcd", "ab") as fh:                                  # half a frame left by a crash
        fh.write(b"\0" * 100)
    assert dcd.DCDWriter(tmp_path / "t.dcd", 7, 10, 0.01).next_step() == 20


def test_set_engine_rewrites_an_unstarted_folder(molecules, tmp_path):
    from quantabench.md import sim

    tm = molecules["c6f14"]
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=25.5, packing="lattice", dispersion="cutoff12",
                             dump_every=0)
    L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.6)
    assert sim.Run.load(tmp_path).engine == "lammps"
    run = L.set_engine(tmp_path, "openmm", kb_every=50, frame_every=500)
    assert run.engine == "openmm" and run.variant == "" and any("OpenMM" in n for n in run.notes)
    bulk, slab = run.protocol.parts
    assert bulk.input == slab.input == "liquid_setup.json" and slab.restart_files == ["slab.state.a.xml",
                                                                                      "slab.state.b.xml"]
    kinds = {o.name: (o.kind, o.every, o.rewrites_start) for o in slab.outputs}
    assert kinds["slab.dcd"] == ("dcd", 500, True) and kinds["sigma_kb.txt"] == ("ave/time", 50, True)
    assert "pressure_tensor.txt" not in kinds and "test_area.txt" not in kinds
    assert run.protocol.pause_every % 500 == 0 and run.protocol.pause_every % 50 == 0
    _tm, cfg2, _meta = L.load_folder(tmp_path)
    assert cfg2.engine == "openmm" and cfg2.kb_every == 50
    assert "ENGINE: OpenMM" in (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert run.parameterization.physics["engine"]["name"] == "openmm"
    run.segments.append(sim.Segment(n=1, part="bulk", status="paused"))
    run.save(tmp_path)
    with pytest.raises(sim.RunError, match="has started"):
        L.set_engine(tmp_path, "lammps")
    with pytest.raises(ValueError):
        L.LiquidSlabConfig(engine="gromacs").resolved()


def test_benchmark_variants_and_recommendation_know_openmm():
    from quantabench.md import bench
    from quantabench.md import long_range as LR

    assert bench.parse_variant("openmm-gpu").name == "openmm-gpu-cutoff16"
    assert bench.parse_variants("default,openmm")[-1] == "openmm-gpu-cutoff16"
    assert bench.parse_variant("OpenMM-CPU-ewald").name == "openmm-cpu-ewald"
    with pytest.raises(ValueError):
        bench.parse_variant("openmm-tpu-cutoff16")
    recs = {"variants": {"openmm-gpu-cutoff16@slab": {"ns_per_day": 90.0, "atoms": 9780, "accelerator": "openmm",
                                                       "platform": "OpenCL", "precision": "single", "check": "ok"},
                         "cpu-ewald@slab": {"ns_per_day": 7.0, "atoms": 9780, "threads": 8}}}
    est = bench.estimate_hours(recs, 9780, 12.0, dispersion="cutoff16")
    assert est[0].variant == "openmm-gpu-cutoff16" and est[0].hours == pytest.approx(12.0 / 90.0 * 24.0)
    assert "OpenMM OpenCL single" in " ".join(bench.describe_lines(recs))
    ok, detail = bench.check_energy({"pe": 497.40, "evdwl": -2234.4, "ecoul": 5000.0, "elong": 1400.0},
                                    {"pe": 497.80})
    assert ok == "ok" and "497.800" in detail
    m = LR.MachineInfo(True, "test", engine="openmm", benchmark=recs)
    rec = LR.recommend_long_range("slab", (40.3, 40.3, 200.9), n_atoms=9780, machine=m)
    assert (rec.method, rec.cutoff, rec.accelerator) == ("cutoff", 16.0, "gpu") and "OpenMM" in rec.reason
    # pilot B's σ as MD-LR-2 corrected it (11.71 as run), and what A-fine shows, as in LAMMPS' reason
    assert "11.41 ± 0.30 vs 11.25" in rec.reason and "11.71" not in rec.reason
    assert "9.6 % below" in rec.reason and "12.62" in rec.reason
    assert rec.ns_per_day == pytest.approx(90.0) and rec.speed_source == "benchmark"
    ref = LR.recommend_long_range("slab", (40.3, 40.3, 200.9), n_atoms=9780, machine=m, purpose="reference")
    assert (ref.method, ref.accelerator) == ("ewald", "gpu") and "LJ-PME" in ref.reason
    # GPU-SCREEN-1: the model is checked against A-fine; the mixed-precision on-the-fly samples are not usable as they are
    assert any("0.3 mN/m high in mixed precision" in n for n in ref.notes)
    assert not any("has not yet been checked" in n for n in ref.notes)
    assert not any("GPU stays idle" in n for n in ref.notes)


def test_ljpme_mesh_is_always_given():
    """GPU-SCREEN-1: LJ-PME gets a mesh even when the folder names none (``disp_grid`` "accuracy", or a folder written
    before MD-LR): OpenMM given a G and a zero mesh computes −inf on its CPU platform and crashes on Reference. The
    rule (G 2.77 / rc, h ≤ 0.52 / G) gives A-fine's own 18 x 18 x 90 for its slab box."""
    g, n = OE.ljpme_parameters(OE.Physics(lj="pme", cutoff_A=12.0), (40.4457, 40.4457, 201.187))
    assert g == pytest.approx(2.77 / 12.0) and n == (18, 18, 90)
    assert OE.ljpme_parameters(OE.Physics(lj="pme", disp_gewald=0.2717, disp_spacing=1.0), (40.0, 40.0, 200.0)) == (
        0.2717, (40, 40, 200))
    meta = {"molecule": {}, "config": {"dispersion": "ewald", "cutoff": 12.0, "disp_grid": "accuracy",
                                       "kspace_accuracy": 1e-6}}
    phys = OE.physics_from_setup(meta, "slab")
    assert phys.lj == "pme" and phys.disp_spacing is None and phys.ewald_tolerance == 1e-5
    assert OE.ljpme_parameters(phys, (40.0, 40.0, 200.0))[1] == (18, 18, 89)


def _openmm_like_folder(folder: Path, kb_bias: float = 0.0, n: int = 40_000, every: int = 20, frame_every: int = 1000,
                        seed: int = 2) -> None:
    """A slab folder as an OpenMM run leaves it: slab_box.txt, sigma_kb.txt (σ samples every ``every`` steps),
    pressure_tensor.txt at the frames (the rerun: the same σ, ``kb_bias`` below the on-the-fly samples), run.json."""
    import json

    rng = np.random.default_rng(seed)
    lz = 200.0
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "slab_box.txt").write_text(f"QB_SLAB lx 40 ly 40 lz {lz} zlo 0 zhi {lz} dt 1 p_every {every} "
                                         f"settle_steps 0 production_steps {n * every} T 298.15 rc 16 bulk_lz 100 "
                                         "engine openmm\n", encoding="utf-8")
    steps = np.arange(0, n * every, every)
    sig = 12.0 + 300.0 * rng.normal(size=len(steps))
    np.savetxt(folder / "sigma_kb.txt", np.column_stack([steps, sig, np.zeros_like(sig)]),
               header="TimeStep sigma_mN_m dU_kcal", fmt="%.8g")
    fs = steps[steps % frame_every == 0]
    s_frames = sig[steps % frame_every == 0] - kb_bias + 0.05 * rng.normal(size=len(fs))
    pzz = s_frames / (lz / 2 * 0.0101325)                     # Pxx = Pyy = 0: σ = (Lz/2) Pzz
    np.savetxt(folder / "pressure_tensor.txt", np.column_stack([fs, 0 * fs, 0 * fs, pzz]), fmt="%.10g",
               header="TimeStep c_thermo_press[1] c_thermo_press[2] c_thermo_press[3]")
    (folder / "run.json").write_text(json.dumps({"engine": "openmm", "segments": []}), encoding="utf-8")


def test_analyze_takes_the_checked_on_the_fly_samples(tmp_path):
    from quantabench.md import surface_tension as S

    _openmm_like_folder(tmp_path / "ok")
    r = S.analyze(tmp_path / "ok", skip_ns=0.0, block_ns=0.1)
    s = r["sigma"]
    assert r["engine"] == "openmm" and s["source"].startswith("on the fly") and s["check"]["ok"]
    assert s["n_samples"] == 40_000 and s["sample_every_fs"] == 20 and abs(s["mean_mN_m"] - 12.0) < 5 * s["sem_mN_m"]
    assert r["sigma_rerun"]["n_samples"] == 800 and r["sigma_rerun"]["sem_mN_m"] > 2 * s["sem_mN_m"]
    assert "on the fly − rerun" in S.format_report(r)
    _openmm_like_folder(tmp_path / "biased", kb_bias=1.0)
    r = S.analyze(tmp_path / "biased", skip_ns=0.0, block_ns=0.1)
    assert r["sigma"]["source"].startswith("rerun") and r["sigma_on_the_fly"]["check"]["ok"] is False
    assert any(not c["ok"] and "on-the-fly" in c["check"] for c in r["checks"])


def _with_frames(folder: Path, frame_every: int = 1000, n_frames: int = 800, keep: int | None = None) -> None:
    """slab.dcd (3 atoms) at the steps of ``_openmm_like_folder``'s rerun rows, and liquid_setup.json."""
    import json

    from quantabench.md import dcd

    w = dcd.DCDWriter(folder / "slab.dcd", 3, frame_every, frame_every / 1000.0)
    for k in range(n_frames):
        w.write(k * frame_every, np.full((3, 3), float(k)), (40.0, 40.0, 200.0))
    cfg = {"frame_every": frame_every, **({} if keep is None else {"frame_keep_every": keep})}
    (folder / "liquid_setup.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")


def test_frames_are_thinned_only_after_the_check_passes(tmp_path, monkeypatch, capsys):
    """Brandon, 2026-11-24: once the frames are rerun and the on-the-fly σ has passed its check, every 10th frame
    is kept (frame_keep_every) and the rest deleted; never before (slab part running, frames not rerun, check
    failed), and pressure_tensor.txt keeps every row, so the analysis is unchanged."""
    from types import SimpleNamespace

    from quantabench import cli
    from quantabench.md import dcd, sim
    from quantabench.md import openmm_sigma as OS
    from quantabench.md import surface_tension as S

    state = {"done": False}
    monkeypatch.setattr(sim.Run, "load", staticmethod(lambda f: SimpleNamespace(part_done=lambda p: state["done"])))
    ok = tmp_path / "ok"
    _openmm_like_folder(ok)
    _with_frames(ok)
    r = S.analyze(ok, skip_ns=0.0, block_ns=0.1)
    chk = r["sigma"]["check"]
    assert chk["ok"] and L.LiquidSlabConfig().frame_keep_every == 10
    size = (ok / "slab.dcd").stat().st_size
    out = OS.thin_frames(ok, chk)                                       # the slab part is still running
    assert not out["thinned"] and "not finished" in out["why"] and (ok / "slab.dcd").stat().st_size == size
    state["done"] = True
    assert "not passed" in OS.thin_frames(ok, None)["why"]
    assert "not passed" in OS.thin_frames(ok, {**chk, "ok": False})["why"]
    assert "every frame stays" in OS.thin_frames(ok, chk, keep_every=0)["why"]
    w = dcd.DCDWriter(ok / "slab.dcd", 3, 1000, 1.0)                    # a frame without its rerun row
    w.write(800_000, np.zeros((3, 3)), (40.0, 40.0, 200.0))
    assert "1 frames are not in" in OS.thin_frames(ok, chk)["why"]
    dcd.truncate(ok / "slab.dcd", 799_000, keep_step=True)
    assert dcd.frame_steps(ok / "slab.dcd")[-1] == 799_000 and not (ok / OS.THINNED_FILE).exists()

    args = ["md", "surface-tension", str(ok), "--skip-ns", "0", "--block-ns", "0.1", "--rerun", "none"]
    assert cli.main([*args, "--keep-frames"]) == 0
    assert len(dcd.frame_steps(ok / "slab.dcd")) == 800
    assert cli.main(args) == 0
    assert "Kept every 10th frame of slab.dcd (800 -> 80 frames" in capsys.readouterr().err
    steps = dcd.frame_steps(ok / "slab.dcd")
    assert steps == list(range(0, 800_000, 10_000)) and dcd.read_header(ok / "slab.dcd").every == 10_000
    got = list(dcd.read_frames(ok / "slab.dcd"))
    assert got[3][0] == 30_000 and np.all(got[3][1] == 30.0)          # the frames kept are the ones at those steps
    rec = OS.thinned(ok)
    assert (rec["frames_before"], rec["frames_after"], rec["last_step"]) == (800, 80, 799_000)
    assert rec["bytes_after"] < rec["bytes_before"] / 9 and rec["check"]["ok"]
    assert "every 10th kept after the check passed: 800 → 80" in (ok / "surface_tension.md").read_text(encoding="utf-8")
    assert len(OS.done_steps(ok)) == 800 and OS.pending_frames(ok) == 0
    assert OS.rerun(ok, "openmm")["method"] == "none" and len(OS.done_steps(ok)) == 800   # its rows stay
    again = S.analyze(ok, skip_ns=0.0, block_ns=0.1)
    assert again["sigma"]["mean_mN_m"] == r["sigma"]["mean_mN_m"] and again["sigma_rerun"]["n_samples"] == 800
    assert OS.thin_frames(ok, chk)["why"] == "thinned earlier" and dcd.frame_steps(ok / "slab.dcd") == steps
    with pytest.raises(ValueError, match="in the file"):
        dcd.DCDWriter(ok / "slab.dcd", 3, 1000, 1.0)                    # a thinned file is not continued

    biased = tmp_path / "biased"                                        # the check fails: every frame stays
    _openmm_like_folder(biased, kb_bias=1.0)
    _with_frames(biased)
    assert cli.main(["md", "surface-tension", str(biased), "--skip-ns", "0", "--block-ns", "0.1", "--rerun",
                     "none"]) == 0
    assert len(dcd.frame_steps(biased / "slab.dcd")) == 800 and not (biased / OS.THINNED_FILE).exists()
    off = tmp_path / "off"
    _openmm_like_folder(off)
    _with_frames(off, keep=0)
    assert cli.main(["md", "surface-tension", str(off), "--skip-ns", "0", "--block-ns", "0.1", "--rerun",
                     "none"]) == 0
    assert len(dcd.frame_steps(off / "slab.dcd")) == 800


# --------------------------------------------------------------------------------------- a real OpenMM run
def _tiny(**kw):
    base = dict(lateral=18.0, slab_thickness=18.0, packing="lattice", dispersion="cutoff8", thermo_every=100,
                warmup_ps=0.2, compress_ps=0.5, bulk_ns=0.0013, settle_ns=0.0005, production_ns=0.0025,
                pressure_every=10, profile_every=10, profile_repeat=10, profile_freq=100, restart_every=1000,
                dump_every=0, vacuum=20.0, engine="openmm", frame_every=100, kb_every=20, ta_every=50)
    base.update(kw)
    return L.LiquidSlabConfig(**base)


def test_openmm_run_pauses_resumes_and_gives_sigma(omm_python, tmp_path):
    """A tiny hexane slab through the run supervisor on OpenMM's CPU platform: a pause in the slab and a resume give
    one series of frames, on-the-fly σ samples and density profiles; md surface-tension reruns the frames (LAMMPS
    where it is found, else OpenMM's CPU platform) and checks the on-the-fly samples against them."""
    from quantabench import cli
    from quantabench.md import dcd, sim
    from quantabench.md import supervisor as SV
    from quantabench.md import surface_tension as S

    folder = tmp_path / "hexane"
    tm = F.type_molecule("CCCCCC", "hexane", "oplsaa")
    L.write_liquid_slab(tm, folder, _tiny(), density_estimate=0.66)

    def pause_once():
        p = sim.read_json(folder / sim.PROGRESS_FILE) or {}
        if p.get("part") == "slab" and (p.get("step") or 0) >= 500 and not (folder / "paused.flag").exists():
            (folder / "STOP").write_text("pause", encoding="utf-8")
            (folder / "paused.flag").write_text("", encoding="utf-8")
        return False

    run = SV.Supervisor(folder, variant="cpu", threads=1, priority="low", echo=None, progress_every_s=0.1,
                        should_stop=pause_once).run()
    assert run.state == "paused", [s.__dict__ for s in run.segments]
    seg = run.segments[-1]
    assert seg.part == "slab" and seg.end["kind"] == "xml" and seg.end["step"] % run.protocol.pause_every == 0
    assert seg.variant == "openmm-cpu-mixed" and seg.build["engine"] == "openmm" and seg.ns_per_day > 0
    run = SV.Supervisor(folder, threads=1, priority="low", echo=None, resume=True).run()
    assert run.state == "done" and run.segments[-1].start["kind"] == "xml"
    end = run.protocol.parts[1].last_step
    kb = np.loadtxt(folder / "sigma_kb.txt", comments="#")
    assert kb[0, 0] == 0 and kb[-1, 0] == end and np.all(np.diff(kb[:, 0]) == 20)
    assert dcd.frame_steps(folder / "slab.dcd") == list(range(0, end + 1, 100))
    steps, _z, _rho = S.read_profiles(folder / "density_profile.txt")
    assert list(steps) == list(range(100, end + 1, 100))
    b = S.read_bulk_density(folder / "bulk_density.txt")
    assert len(b) >= 1 and 0.3 < b[-1, 1] < 1.2
    for name in ("bulk_final.data", "slab_start.data", "slab_final.data", "slab_final.xml", "slab_box.txt"):
        assert (folder / name).is_file(), name
    assert cli.main(["md", "surface-tension", str(folder), "--skip-ns", "0.0002", "--threads", "1"]) == 0
    r = S.analyze(folder, skip_ns=0.0002)
    assert r["engine"] == "openmm" and np.isfinite(r["sigma"]["mean_mN_m"])
    chk = r["sigma"].get("check") or (r.get("sigma_on_the_fly") or {}).get("check")
    assert chk and chk["n"] >= 20
    # LAMMPS' virial (Coulomb k-space at 1e-6) and OpenMM's strain derivative of the same frames agree closely
    assert abs(chk["mean_diff_mN_m"]) <= 0.1 + 3 * chk["sem_diff_mN_m"], chk     # per-frame noise: the CPU platform
    assert "sigma_test_area" in r
    from quantabench.md import openmm_sigma as OS

    if chk["ok"]:                     # every 10th frame kept once the check passed; the analysis is unchanged
        assert dcd.frame_steps(folder / "slab.dcd") == list(range(0, end + 1, 1000)) and OS.thinned(folder)
        assert OS.pending_frames(folder) == 0 and S.analyze(folder, skip_ns=0.0002)["sigma"] == r["sigma"]
    else:
        assert dcd.frame_steps(folder / "slab.dcd") == list(range(0, end + 1, 100))


def test_settings_engines_tab_has_the_md_engine_and_its_note(tmp_path):
    """Settings > Engines > Molecular dynamics: the MD engine (Automatic by default) with the note on what OpenMM
    gives, saved for this computer only (Brandon, 2026-11-24), and the OpenMM Python path among the programs."""
    from conftest import HAS_DISPLAY

    if not HAS_DISPLAY:
        pytest.skip("needs a display")
    from PySide6.QtWidgets import QApplication, QLabel

    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = QApplication.instance() or QApplication([])
    s = sm.Settings(output_dir=str(tmp_path))
    s.engines.machine_md_engine = {"other-pc": "lammps"}
    d = SettingsDialog(s, None)
    try:
        assert d.md_engine.currentData() == "auto" and "openmm_python" in d.program_paths
        note = d.findChild(QLabel, "md_engine_note")
        assert note is not None and note.text() == L.ENGINE_NOTE and "single precision on Apple GPUs" in note.text()
        d.md_engine.setCurrentIndex(d.md_engine.findData("openmm"))
        out = d.result_settings()
        assert out.engines.md_engine == "auto"                         # the shared default is not changed
        assert out.md_engine_for() == "openmm" and out.md_engine_for("other-pc") == "lammps"
        assert "engines.machine_md_engine" in sm._machine_entry(out.dist.machine_updated, None)
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()


def test_md_engine_is_set_per_computer_and_automatic_by_default(tmp_path, monkeypatch):
    """engines.machine_md_engine: one entry per computer, which wins over engines.md_engine ("auto" by default:
    OpenMM where OpenMM sees a usable GPU, else LAMMPS); settings.save keeps the other computers' entries."""
    import quantabench.settings as settings_mod
    from quantabench import cli_md
    from quantabench.md import openmm_engine as OE

    s = settings_mod.Settings(output_dir=str(tmp_path))
    assert s.engines.md_engine == "auto" and s.md_engine_for(scope=True) == ("auto", "all machines")
    s.set_md_engine_for("LAMMPS", "Lab PC")
    assert s.engines.machine_md_engine == {"lab-pc": "lammps"}
    assert s.md_engine_for("lab_pc", scope=True) == ("lammps", "this machine") and s.md_engine_for("mac") == "auto"
    s.engines.md_engine = "openmm"
    assert s.md_engine_for("mac", scope=True) == ("openmm", "all machines") and s.md_engine_for("lab-pc") == "lammps"
    s.engines.machine_md_engine["mac"] = "fast"                       # not an engine: the shared one
    assert s.md_engine_for("mac") == "openmm"
    s.set_md_engine_for("", "lab-pc")
    assert "lab-pc" not in s.engines.machine_md_engine
    with pytest.raises(ValueError, match="MD engine"):
        s.set_md_engine_for("gromacs")
    assert ("engines", "machine_md_engine") in settings_mod.PER_MACHINE_TABLES      # merged entry by entry on save
    back = settings_mod.Settings.from_dict(s.to_dict())
    assert back.engines.machine_md_engine == s.engines.machine_md_engine

    gpu = {"platforms": [{"name": "CPU", "precisions": ["mixed"], "devices": []},
                         {"name": "OpenCL", "precisions": ["single"], "devices": ["Apple M3 Max"]}]}
    cpu = {"platforms": [{"name": "Reference", "precisions": ["double"]}, {"name": "CPU", "precisions": ["mixed"]}]}
    assert OE.usable_gpu(gpu) == "OpenCL on Apple M3 Max (single)" and OE.usable_gpu(cpu) == "" == OE.usable_gpu({})
    s = settings_mod.Settings(output_dir=str(tmp_path))
    monkeypatch.setattr(settings_mod, "load", lambda *a, **k: s)
    for desc, engine in ((gpu, "openmm"), (cpu, "lammps")):
        monkeypatch.setattr(OE, "probe_platforms", lambda *a, _d=desc, **k: _d)
        got, why = cli_md.default_engine(with_reason=True)
        assert got == engine and why.startswith("automatic") and cli_md.default_engine() == engine
    s.set_md_engine_for("lammps")                                     # this computer's own choice wins
    assert cli_md.default_engine(with_reason=True) == ("lammps", "set for this machine")
