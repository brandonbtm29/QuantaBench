"""MD-BENCH-2: the slab benchmark box, its inputs and records (md.bench), the LAMMPS it refuses to time, how a LAMMPS
that was stopped is reported, the geometry-aware estimates and their scaling rule, and measured speeds in the
long-range recommendation (md.long_range). A real slab benchmark runs at the end (skipped without LAMMPS)."""

import json
import math

import pytest
from test_md_bench import HELP_GPU, HELP_WHEEL, lammps_info

from quantabench import proctree
from quantabench.chem import external
from quantabench.dist import profile
from quantabench.dist.profile import MachineProfile
from quantabench.md import bench
from quantabench.md import forcefield as F
from quantabench.md import liquid as L
from quantabench.md import long_range as LR

PILOT_SLAB = (40.33, 40.33, 200.90)
#: the macOS PyPI wheel (abridged): OPENMP built without OpenMP
HELP_MAC_WHEEL = HELP_WHEEL.replace("OPENMP package API: OpenMP", "OPENMP package API: Serial")

LOG_SLAB = """
PPPMDisp initialization ...
  Coulomb G vector (1/distance)= 0.2913746
  Coulomb grid = 24 24 45
  Coulomb stencil order = 5
  Coulomb estimated absolute RMS force accuracy = 0.00032
  Dispersion G vector (1/distance)= 0.230833
  Dispersion grid = 18 18 36
  Dispersion stencil order = 5
QB_TIMED
Loop time of 30.01 on 8 procs for 4000 steps with 3800 atoms

Performance: 11.517 ns/day, 2.084 hours/ns, 133.289 timesteps/s, 506.5 katom-step/s
799.0% CPU use with 1 MPI tasks x 8 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 18.2       | 18.2       | 18.2       |   0.0 | 60.65
Bond    | 1.5        | 1.5        | 1.5        |   0.0 |  5.00
Kspace  | 7.5        | 7.5        | 7.5        |   0.0 | 24.99
Neigh   | 1.8        | 1.8        | 1.8        |   0.0 |  6.00
Comm    | 0.6        | 0.6        | 0.6        |   0.0 |  2.00
Output  | 0.1        | 0.1        | 0.1        |   0.0 |  0.33
Modify  | 0.3        | 0.3        | 0.3        |   0.0 |  1.00
Other   |            | 0.01       |            |       |  0.03

Thread timing breakdown (MPI rank 0):
Total threaded time 0:00:30 / 99.0%
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 17.9       | 18.0       | 18.1       |   0.2 | 59.00
"""


@pytest.fixture(scope="module")
def c6f14():
    return F.type_molecule(bench.SMILES, bench.MOLECULE, bench.FORCEFIELD)


def slab_rec(ns, atoms=3800, box=(40.0, 40.0, 78.0), kspace=None, **kw):
    rec = {"ns_per_day": ns, "atoms": atoms, "threads": 8, "accelerator": "omp", "geometry": "slab",
           "box_A": list(box), "date": "2026-11-24T23:30:00+00:00", "long_range": "(the settings it timed)"}
    if kspace is not None:
        rec["breakdown"] = {"Pair": 100.0 - kspace, "Kspace": kspace}
    rec.update(kw)
    return rec


def md(**variants):
    return {"bench_version": 2, "date": "2026-11-24T23:30:00+00:00", "lammps": "4 Jul 2026", "variants": variants}


# ------------------------------------------------------------------------------------------- sizes and keys
def test_sizes_geometries_and_keys():
    assert bench.DEFAULT_SIZE == "slab" and bench.MD_BENCH_VERSION == 4     # 4: cutoff variants' Coulomb mesh rule
    assert "MD-LR-2" in bench.VERSION_CHANGES[4] and "OPENMM-1" in bench.VERSION_CHANGES[3]
    assert {k: bench.geometry_of(k) for k in bench.SYSTEMS} == {"slab": "slab", "small": "bulk",
                                                                 "pilot-slab": "slab", "pilot": "bulk"}
    assert bench.SYSTEMS["slab"]["config"]["n_molecules"] * 20 == 3800
    assert bench.SYSTEMS["slab"]["config"]["lateral"] > 2 * 16 + 1        # cutoff16 fits (md.liquid.plan_box)
    with pytest.raises(ValueError, match="size"):
        bench.geometry_of("huge")
    assert bench.record_key("cpu-ewald", "slab") == "cpu-ewald@slab" and bench.record_key("cpu-ewald") == "cpu-ewald"
    assert bench.split_key("gpu-cutoff16@slab") == ("gpu-cutoff16", "slab")
    assert bench.split_key("gpu-cutoff16") == ("gpu-cutoff16", "bulk")
    assert bench.split_key("x@weird") == ("x@weird", "bulk")
    keys = ["gpu-cutoff16@slab", "cpu-ewald", "cpu-cutoff12@slab", "cpu-ewald@slab", "junk"]
    assert sorted(keys, key=bench._order) == ["cpu-ewald@slab", "cpu-ewald", "cpu-cutoff12@slab",
                                              "gpu-cutoff16@slab", "junk"]
    assert bench._vacuum_A("slab", 39.0) == 19.5 and bench._vacuum_A("pilot-slab", 100.0) == 50.0
    assert bench._vacuum_A("small", 37.0) == 0.0
    # the slab box keeps the pilot's box / liquid ratio: its dispersion mesh per atom is the pilot's
    h = LR.rule_spacing(LR.rule_gewald(12.0))
    small, pilot = LR.disp_mesh((40.0, 40.0, 78.0), h), LR.disp_mesh(PILOT_SLAB, h)
    assert small == (18, 18, 36) and pilot == (18, 18, 90)
    assert math.prod(small) / 3800 == pytest.approx(math.prod(pilot) / 9780, rel=0.05)


def test_slab_inputs_time_what_a_surface_tension_run_does(c6f14):
    slab = bench.bench_input(c6f14, "cutoff16", steps=1000, geometry="slab", label="cpu-cutoff16")
    bulk = bench.bench_input(c6f14, "cutoff16", steps=1000)
    assert "tail yes" in bulk and "tail yes" not in slab and "NO tail correction" in slab
    for line in ("fix             nodrift all momentum 1000 linear 1 1 1", "file bench_pressure.txt",
                 "compute         zbins all chunk/atom bin/1d z lower 0.5 units box", "file bench_profile.txt"):
        assert line in slab and line not in bulk
    assert "fix             ptens all ave/time 10 1 10 c_thermo_press[1]" in slab
    assert slab.index("run             0 post no") < slab.index('print           "QB_TIMED"')
    # MD-LR-2's Coulomb mesh rule, as in md liquid-slab's inputs (benchmark version 4): the cutoff variants set
    # G and one mesh spacing from the box; pppm/disp (ewald) picks its own
    for text in (slab, bulk):
        assert "variable coul_g         equal" in text and text.count("kspace_modify   gewald $(v_coul_g") == 1
    ewald = bench.bench_input(c6f14, "ewald", steps=1000, geometry="slab", label="cpu-ewald")
    assert "coul_g" not in ewald and "pppm/disp" in ewald
    ewald = bench.bench_input(c6f14, "ewald", steps=10, geometry="slab")
    assert "kspace_style    pppm/disp ${kacc}" in ewald and "mesh/disp $(ceil(lx/v_disp_h)" in ewald
    assert "variable kacc           index 1e-06" in ewald and "G 0.2308 1/A" in ewald
    with pytest.raises(ValueError):
        bench.bench_input(c6f14, "ewald", steps=10, geometry="droplet")


def test_prep_opens_the_slab_as_in_slab_does(c6f14):
    bulk = bench._prep_input(c6f14, (37.0, 37.0, 37.0))
    slab = bench._prep_input(c6f14, (40.0, 40.0, 39.0), vacuum=19.5)
    opening = "\n".join(L.open_box_lines())
    assert opening in slab and opening not in bulk and "variable        vacuum index 19.500000" in slab
    assert f"run             {bench.SLAB_RELAX_STEPS}" in slab and "liquid_lz $(v_lz0:%.6f)" in bulk + slab
    assert slab.index("change_box") < slab.index("write_data      bench_start.data")
    # in.slab is unchanged by the refactor: the same lines, then the dispersion mesh again
    cfg = L.LiquidSlabConfig(lateral=40.0, n_molecules=190).resolved()
    text = L.slab_input(c6f14, cfg, ["C", "F"])
    assert opening in text and text.index(opening) < text.index("the dispersion mesh again from disp_h")


# ------------------------------------------------------------------------------------------- logs and exits
def test_kspace_grids_and_timing_breakdown_from_the_log():
    k = bench.parse_kspace(LOG_SLAB)
    assert k == {"coulomb_g": 0.2913746, "coulomb_grid": [24, 24, 45], "dispersion_g": 0.230833,
                 "dispersion_grid": [18, 18, 36]}
    plain = "PPPM initialization ...\n  G vector (1/distance) = 0.25161\n  grid = 20 20 36\n  stencil order = 5\n"
    assert bench.parse_kspace(plain) == {"coulomb_g": 0.25161, "coulomb_grid": [20, 20, 36]}
    assert bench.parse_kspace("no kspace") == {}
    b = bench.parse_breakdown(LOG_SLAB)
    assert b["Kspace"] == 24.99 and b["Pair"] == 60.65 and b["Other"] == 0.03     # the MPI table, not the threads'
    assert len(b) == 8 and bench.parse_breakdown("Loop time of 1 on 1 procs") == {}


def test_exit_reason_names_the_signal():
    assert bench.exit_reason(0) == bench.exit_reason(1) == bench.exit_reason(None) == ""
    for code in (-15, 143, 241):          # Python, a shell, PyPI's lmp wrapper (sys.exit(-15))
        why = bench.exit_reason(code)
        assert why.startswith("killed by SIGTERM (signal 15)") and "outside QuantaBench" in why, code
    assert "128 + 15" in bench.exit_reason(143) and "PyPI" in bench.exit_reason(241)
    assert "out of memory" in bench.exit_reason(137) and "crashed" in bench.exit_reason(-11)
    assert bench.exit_reason(137).startswith("killed by SIGKILL (signal 9)")    # on Windows too (POSIX_SIGNALS)
    assert bench.POSIX_SIGNALS[15] == "SIGTERM" and bench.exit_reason(-99).startswith("killed by signal 99")


def test_prepare_system_keeps_a_failed_set_up_and_says_why(monkeypatch):
    lmp = lammps_info(HELP_WHEEL)
    monkeypatch.setattr(L, "find_packmol", lambda: None)               # a lattice start: no Packmol needed
    monkeypatch.setitem(bench.SYSTEMS, "tiny-slab", {"config": {"lateral": 26.0, "n_molecules": 40},
                                                    "vacuum": "half"})

    def stopped(cmd, cwd, **kw):                    # what the MacBook saw on 2026-11-21: exit 143 after ~2 min
        (cwd / kw["log_name"]).write_text("LAMMPS (22 Jul 2025 - Update 4)\n", encoding="utf-8")
        return external.ProgramRun(143, 118.0, cwd / kw["log_name"])

    monkeypatch.setattr(external, "run_program", stopped)
    with pytest.raises(RuntimeError) as err:
        bench.prepare_system("tiny-slab", lmp, 2)
    text = str(err.value)
    assert "exited with code 143 after 118 s" in text and "killed by SIGTERM" in text and "failed-tiny-slab-" in text
    kept = [d for d in bench.system_dir("tiny-slab").parent.iterdir() if d.name.startswith("failed-tiny-slab-")]
    assert len(kept) == 1 and (kept[0] / "in.prep").is_file() and (kept[0] / "out.prep").is_file()
    assert "change_box" in (kept[0] / "in.prep").read_text(encoding="utf-8")
    assert not bench.system_dir("tiny-slab").exists()
    rec = bench.run_md_benchmark(["cpu-cutoff12"], size="tiny-slab", threads=2, lammps=lmp)
    assert rec["skipped"].startswith("the tiny-slab benchmark box could not be built: LAMMPS (")
    assert "RuntimeError" not in rec["skipped"] and "SIGTERM" in rec["skipped"]


# ------------------------------------------------------------------------------------------- which LAMMPS
def test_a_bad_explicit_lammps_is_refused_not_passed_over(monkeypatch, tmp_path):
    """The MacBook, 2026-11-21: QUANTABENCH_LAMMPS lost its backslashes on its way through PowerShell and SSH
    ($HOME expanded on Windows), pointed at nothing, and the benchmark timed the venv's PyPI wheel instead."""
    mangled = "C:Usersyou/miniforge3/envs/qb-lammps/bin/lmp"
    monkeypatch.setenv("QUANTABENCH_LAMMPS", mangled)
    assert external.configured_problems("lammps") == [
        f"QUANTABENCH_LAMMPS is set to {mangled!r}, which does not exist"]
    wheel = external.ProgramInfo("lammps", "/repo/.venv/bin/lmp",
                                 f"this environment (environment variable: {mangled} not found)")
    monkeypatch.setattr(external, "find_program", lambda *a, **k: wheel)
    monkeypatch.setattr(proctree, "run_probe", lambda cmd, timeout=15.0, **k: proctree.Probe(0, HELP_GPU))
    info, why = bench.find_lammps()
    assert info is None and "QUANTABENCH_LAMMPS is set to" in why and "/repo/.venv/bin/lmp (this environment)" in why
    assert "--md-lammps" in why
    rec = bench.run_md_benchmark()
    assert rec["skipped"] == why and rec["variants"] == {}
    # --md-lammps: that file, whatever the variable says
    exe = tmp_path / "lmp"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    info, why = bench.find_lammps(path=str(exe))
    assert info.path == str(exe) and info.source == "given (--md-lammps)" and info.has("GPU") and not why
    assert "no such file" in bench.find_lammps(path=str(tmp_path / "missing"))[1]
    # a variable that is right wins (find_program's source says so); unset, nothing is refused
    monkeypatch.setenv("QUANTABENCH_LAMMPS", str(exe))
    assert external.configured_problems("lammps") == []
    monkeypatch.delenv("QUANTABENCH_LAMMPS")
    assert bench.find_lammps()[0].path == "/repo/.venv/bin/lmp"


def test_this_machines_settings_path_counts_the_shared_one_does_not(monkeypatch):
    from quantabench import settings as settings_mod

    monkeypatch.delenv("QUANTABENCH_LAMMPS", raising=False)
    s = settings_mod.load()
    s.engines.paths["lammps"] = r"C:\Program Files\LAMMPS 64-bit 4Jul2026\bin\lmp.exe"   # another computer's
    settings_mod.save(s)
    assert external.configured_problems("lammps") == []           # it may be that computer's file: not refused
    s.set_engine_path_for("lammps", "/no/such/lmp")
    settings_mod.save(s)
    problems = external.configured_problems("lammps")
    assert problems == ["this computer's lammps path in the settings is set to '/no/such/lmp', which does not exist"]


def test_an_openmp_build_without_threads_runs_on_one(monkeypatch):
    mac = lammps_info(HELP_MAC_WHEEL)
    assert mac.omp_serial and not lammps_info(HELP_WHEEL).omp_serial and not lammps_info(HELP_GPU).omp_serial
    assert bench.accelerator_args(mac, "cpu", 10) == (["-sf", "omp", "-pk", "omp", "1"], 1, "omp-serial")
    assert "OPENMP: Serial, double" in mac.build_text() and mac.build_text().startswith("22 Jul 2025 - Update 4")
    monkeypatch.setattr(bench, "prepare_system", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    rec = bench.run_md_benchmark(lammps=mac)
    assert any("OPENMP package API: Serial" in n for n in rec["notes"]) and "stop here" in rec["skipped"]


# ------------------------------------------------------------------------------------------- the records
def test_slab_runs_are_keyed_by_geometry_and_kept_beside_bulk(monkeypatch, tmp_path):
    lmp = lammps_info(HELP_GPU)
    meta = {"size": "slab", "geometry": "slab", "atoms": 3800, "molecules": 190, "box_A": [40.0, 40.0, 78.2],
            "liquid_lz_A": 39.1, "vacuum_A": 19.55}
    monkeypatch.setattr(bench, "prepare_system", lambda *a, **k: meta)
    monkeypatch.setattr(bench, "system_dir", lambda size: tmp_path)
    (tmp_path / "molecule.json").write_text(json.dumps(F.type_molecule(bench.SMILES, "C6F14").to_dict()),
                                            encoding="utf-8")
    for f in ("bench_start.data", "ff.params"):
        (tmp_path / f).write_text("", encoding="utf-8")
    seen = []

    def fake_run(lammps, tm, work, v, threads, *, steps=0, geometry="bulk", box=None, tag="", **kw):
        seen.append((tag or v.name, geometry, tuple(box or ())))
        rec = {"run0": {"pe": -100.0, "press": 1.0}, "error": "", "threads": threads, "geometry": geometry}
        if steps:
            rec.update(ns_per_day=10.0 if v.accelerator == "cpu" else 40.0, atoms=3800, steps=1000, seconds=30.0)
        return rec

    monkeypatch.setattr(bench, "run_variant", fake_run)
    rec = bench.run_md_benchmark(threads=4, lammps=lmp)
    assert set(rec["variants"]) == {f"{v}@slab" for v in bench.DEFAULT_VARIANTS}
    assert rec["system"]["geometry"] == "slab" and rec["system"]["vacuum_A"] == 19.55
    assert all(g == "slab" and b == (40.0, 40.0, 78.2) for _n, g, b in seen)
    assert rec["variants"]["gpu-cutoff16@slab"]["check"] == "ok"
    bulk = md(**{"cpu-ewald": {"ns_per_day": 6.5, "atoms": 9780, "threads": 6}})
    both = bench.merge(bulk, rec)
    assert {"cpu-ewald", "cpu-ewald@slab"} <= set(both["variants"])
    line = bench.describe_lines(both)[0]
    assert line.startswith("  md:         cpu-ewald@slab 10 ns/day (3,800 atoms, 4 thr), cpu-ewald 6.5 ns/day")


# ------------------------------------------------------------------------------------------- estimates
def test_scaling_rule_atoms_then_box_volume():
    rec = slab_rec(20.0, kspace=25.0)
    assert bench.scaled_rate(rec, 7600) == (10.0, "")                       # the atom rule
    # same box / liquid ratio: the volume grows with the atoms, both rules agree
    same = bench.scaled_rate(rec, 7600, (40.0, 40.0, 156.0))
    assert same[0] == pytest.approx(10.0) and same[1] == ""
    # a thinner slab in more vacuum: the k-space quarter costs 3 x, the rest 2 x
    rate, how = bench.scaled_rate(rec, 7600, (40.0, 40.0, 3 * 78.0))
    assert rate == pytest.approx(20.0 / (0.75 * 2 + 0.25 * 3)) and "k-space share (25%)" in how
    assert bench.scaled_rate(slab_rec(20.0), 7600, (40.0, 40.0, 234.0))[0] == 10.0   # no breakdown: atoms
    assert bench.scaled_rate({"ns_per_day": 5.0}, 100) == (5.0, "")           # no atoms recorded: unscaled


def test_estimates_prefer_the_jobs_geometry():
    m = md(**{"cpu-ewald@slab": slab_rec(16.0), "cpu-ewald": {"ns_per_day": 20.0, "atoms": 3000, "threads": 8},
              "gpu-cutoff16": {"ns_per_day": 40.0, "atoms": 3000, "threads": 6, "check": "ok"}})
    e = bench.best_estimate(m, 9780, 13.0, dispersion="ewald")                 # slab: the slab record
    assert e.key == "cpu-ewald@slab" and e.geometry == "slab" and e.ns_per_day == pytest.approx(16.0 * 3800 / 9780)
    assert e.note == ""
    e = bench.best_estimate(m, 9780, 13.0, dispersion="ewald", geometry="bulk")
    assert e.key == "cpu-ewald" and e.ns_per_day == pytest.approx(20.0 * 3000 / 9780)
    assert "settings of before MD-LR" in e.note                            # no long_range: the old pppm/disp
    e = bench.best_estimate(m, 9780, 13.0, dispersion="cutoff16")               # the GPU has only a bulk record
    assert e.key == "gpu-cutoff16" and "slab's vacuum mesh was not timed" in e.note
    with pytest.raises(ValueError):
        bench.estimate_hours(m, 9780, 1.0, geometry="droplet")
    lines = bench.estimate_lines(9780, 13.0, "ewald", profiles=[MachineProfile(name="pc", md_throughput=m)])
    assert lines[0].startswith("pc: ") and "(slab box)" in lines[0]


def test_md_estimate_uses_the_folders_slab_box(tmp_path, capsys):
    from quantabench import cli

    m = md(**{"cpu-ewald@slab": slab_rec(16.0, kspace=25.0)})
    profile.save_profile(MachineProfile(name="bench-pc", md_throughput=m, updated="2026-11-24T23:30:00+00:00"))
    out_dir = tmp_path / "c6f14"
    assert cli.main(["md", "liquid-slab", "--smiles", bench.SMILES, "--name", "c6f14", "-o", str(out_dir),
                     "--packing", "lattice", "--lateral", "40", "--molecules", "190", "--long-range", "ewald",
                     "--gpu", "no", "--engine", "lammps"]) == 0
    capsys.readouterr()
    assert cli.main(["md", "estimate", str(out_dir), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    lz = got["box_A"][2]
    assert got["geometry"] == "slab" and got["box_A"][:2] == [40.0, 40.0] and lz > 130     # 2 x 50 Å of vacuum
    row = got["estimates"][0]
    assert row["geometry"] == "slab" and "k-space share" in row["note"]          # more vacuum per atom than 2:1
    assert row["ns_per_day"] < 16.0
    assert cli.main(["md", "estimate", str(out_dir), "--geometry", "bulk"]) == 0
    assert "from the slab benchmark" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- the recommendation
def test_recommendation_uses_this_machines_benchmark():
    m = md(**{"gpu-cutoff16@slab": slab_rec(50.0, accelerator="gpu", threads=6, check="ok"),
              "cpu-ewald@slab": slab_rec(16.0)})
    gpu = LR.recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=LR.MachineInfo(True, "t", benchmark=m))
    assert gpu.speed_source == "benchmark" and gpu.ns_per_day == pytest.approx(50.0 * 3800 / 9780)
    assert gpu.speed_basis.startswith("this machine's MD benchmark: gpu-cutoff16@slab 50 ns/day on 3,800 atoms, "
                                      "GPU + 6 CPU threads, 2026-11-24, scaled to 9,780 atoms")
    assert "this machine's MD benchmark: gpu-cutoff16@slab" in gpu.summary()
    assert any("3.1 x" in n for n in gpu.notes)                                 # 50 against 16 ns/day
    cpu = LR.recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=LR.MachineInfo(False, "t", benchmark=m))
    assert cpu.speed_source == "benchmark" and cpu.ns_per_day == pytest.approx(16.0 * 3800 / 9780)
    by_mol = LR.recommend_long_range("slab", PILOT_SLAB, n_molecules=489, machine=LR.MachineInfo(False, benchmark=m))
    assert by_mol.ns_per_day == pytest.approx(16.0 * 3800 / 9780) and "489 molecules" in by_mol.speed_basis
    # an unverified GPU record is not used; without a fitting record the pilot table speaks, and says so
    bad = md(**{"gpu-cutoff16@slab": slab_rec(50.0, accelerator="gpu", check="mismatch")})
    pilot = LR.recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=LR.MachineInfo(True, benchmark=bad))
    assert pilot.speed_source == "pilot table" and pilot.ns_per_day == pytest.approx(21.0)
    assert "(pilot table: pilot B, GPU + 6 CPU threads)" in pilot.summary()
    # nor an ewald record from before MD-LR: it timed the old, coarser pppm/disp
    old = md(**{"cpu-ewald": {"ns_per_day": 9.0, "atoms": 3000, "threads": 8}})
    assert LR.recommend_long_range("slab", PILOT_SLAB, n_atoms=9780,
                                   machine=LR.MachineInfo(False, benchmark=old)).speed_source == "pilot table"
    # bulk, CPU only: no pilot ran it, the benchmark's bulk record can
    bulk_md = md(**{"cpu-cutoff12": {"ns_per_day": 30.0, "atoms": 3000, "threads": 8}})
    none = LR.recommend_long_range("bulk", machine=False)
    assert none.ns_per_day is None and none.speed_source == "" and "no MD benchmark" in none.speed_basis
    b = LR.recommend_long_range("bulk", (37.0, 37.0, 37.0), n_atoms=3000, machine=LR.MachineInfo(benchmark=bulk_md))
    assert b.speed_source == "benchmark" and b.ns_per_day == pytest.approx(30.0)
    d = gpu.to_dict()
    assert d["speed_source"] == "benchmark" and d["inputs"]["machine"]["benchmark"]["variants"] == [
        "cpu-ewald@slab", "gpu-cutoff16@slab"] and "run0" not in json.dumps(d)


def test_detect_machine_brings_the_stored_benchmark(monkeypatch):
    m = md(**{"cpu-ewald@slab": slab_rec(16.0)})
    profile.save_profile(MachineProfile(name=profile.machine_name(), md_throughput=m))
    monkeypatch.setattr(bench, "find_lammps", lambda refresh=False: (lammps_info(HELP_WHEEL), ""))
    me = LR.detect_machine()
    assert not me.gpu and me.benchmark["variants"]["cpu-ewald@slab"]["ns_per_day"] == 16.0
    assert LR.machine_for("auto").benchmark and not LR.machine_for("no").benchmark   # a stated machine: none
    rec = LR.recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=me)
    assert rec.speed_source == "benchmark"


# ------------------------------------------------------------------------------------------- real LAMMPS
def test_real_slab_benchmark(monkeypatch):
    """A tiny slab (800 atoms, 26 Å lateral, vacuum half its thickness on each side), 200 steps per variant: the box is
    opened and relaxed, the slab inputs run, the dispersion grid follows the slab box (the spacing rule), the records
    carry geometry, box, k-space grids and the timing breakdown, and the CPU/GPU check works in a slab (a "GPU" that
    is the CPU here: the PyPI wheel has no GPU package)."""
    info, why = bench.find_lammps(refresh=True)
    if info is None:
        pytest.skip(why)
    monkeypatch.setitem(bench.SYSTEMS, "tiny-slab", {"config": {"lateral": 26.0, "n_molecules": 40},
                                                    "vacuum": "half"})
    fake_gpu = bench.LammpsInfo(info.path, info.source, info.version, sorted({*info.packages, "GPU"}),
                                {**info.accelerators, "GPU": {"api": "OpenCL", "precision": "mixed"}}, True)
    real_args = bench.accelerator_args
    monkeypatch.setattr(bench, "accelerator_args", lambda lmp, accel, threads: real_args(lmp, "cpu", 1)
                        if accel == "gpu" else real_args(lmp, accel, threads))
    logs = []
    rec = bench.run_md_benchmark(["cpu-ewald", "gpu-cutoff12"], size="tiny-slab", threads=1, max_seconds=20,
                                 steps=200, lammps=fake_gpu, log=logs.append)
    assert not rec["skipped"], logs
    sysm = rec["system"]
    assert sysm["geometry"] == "slab" and sysm["atoms"] == 800 and sysm["vacuum_A"] > 5
    assert sysm["box_A"][2] == pytest.approx(sysm["liquid_lz_A"] + 2 * sysm["vacuum_A"], rel=1e-3)
    assert 1.5 < sysm["density_g_cm3"] < 1.9                                   # the liquid's, not the box's
    v = rec["variants"]
    assert set(v) == {"cpu-ewald@slab", "gpu-cutoff12@slab"}
    for name, r in v.items():
        assert not r["error"] and r["ns_per_day"] > 0 and r["geometry"] == "slab" and r["steps"] == 200, (name, r)
        assert r["box_A"] == sysm["box_A"] and "Kspace" in r["breakdown"] and r["lammps_build"]
    h = LR.rule_spacing(LR.rule_gewald(12.0))
    assert v["cpu-ewald@slab"]["kspace"]["dispersion_grid"] == list(LR.disp_mesh(sysm["box_A"], h))
    assert v["cpu-ewald@slab"]["kspace"]["dispersion_g"] == pytest.approx(LR.rule_gewald(12.0), rel=1e-4)
    assert "for a 26.0 x 26.0" in v["cpu-ewald@slab"]["long_range"]
    assert v["gpu-cutoff12@slab"]["check"] == "ok", v["gpu-cutoff12@slab"]["check_detail"]
    assert any("slab, " in x and "of vacuum on each side" in x for x in logs)
    e = bench.best_estimate(rec, 9780, 1.0, dispersion="ewald", box=PILOT_SLAB)
    assert e.key == "cpu-ewald@slab" and e.ns_per_day > 0
