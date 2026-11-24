"""MD-BENCH-1: the LAMMPS MD benchmark of a machine (md.bench) and its place in the machine registry (dist.profile):
parsing ``lmp -h`` and LAMMPS logs, the CPU/GPU consistency check, the registry record and its display, run-time
estimates, the opt-in wiring of ``machines --bench md`` and the worker, and a real short benchmark (skipped without
LAMMPS)."""

import json
import math
from types import SimpleNamespace

import pytest

from quantabench import cli
from quantabench.dist import profile
from quantabench.dist.profile import MachineProfile
from quantabench.md import bench

#: ``lmp -h`` of the PyPI wheel (abridged): OPENMP, no GPU package
HELP_WHEEL = """
Large-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025 - Update 4
Git info (HEAD / stable_22Jul2025_update4)

Usage example: lmp -var t 300 -echo screen -in in.alloy

-suffix gpu/intel/kk/opt/omp: style suffix to apply (-sf)

Accelerator configuration:

OPENMP package API: OpenMP
OPENMP package precision: double
OpenMP standard: OpenMP 4.5

Installed packages:

AMOEBA ASPHERE BOCS BODY BPM BROWNIAN CG-DNA CG-SPICA CLASS2 COLLOID COLVARS
KSPACE LEPTON MANYBODY MC MOLECULE OPENMP OPT RIGID YAFF

List of individual style options included in this LAMMPS executable

* Atom styles:

amoeba          angle           atomic          body            bond
"""

#: a Windows-installer style build with the GPU package (OpenCL) and KOKKOS on a GPU backend
HELP_GPU = """
Large-scale Atomic/Molecular Massively Parallel Simulator - 4 Jul 2026

Accelerator configuration:

GPU package API: OpenCL
GPU package precision: mixed
KOKKOS package API: HIP Serial
KOKKOS package precision: double
OPENMP package API: OpenMP
OPENMP package precision: double

Compatible GPU present: yes

Installed packages:

GPU KOKKOS KSPACE MANYBODY MOLECULE OPENMP RIGID

"""

LOG_TIMED = """
QB_TIMED
timer           timeout 00:00:15 every 10
New timer settings: style=normal  mode=nosync  timeout=00:00:15
Per MPI rank memory allocation (min/avg/max) = 28.63 | 28.63 | 28.63 Mbytes
   Step          Temp          Press          PotEng
      1100   284.405298    -213.6934906   598.2760175
WARNING: Wall time limit reached (src/src/timer.cpp:201)
Loop time of 14.9732 on 2 procs for 1990 steps with 3000 atoms

Performance: 11.483 ns/day, 2.090 hours/ns, 132.904 timesteps/s, 398.711 katom-step/s
192.4% CPU use with 1 MPI tasks x 2 OpenMP threads
print           "QB_END temp $(temp:%.6g) press $(press:%.6g)"
QB_END temp 301.2 press -12.5
"""

RUN0 = {"pe": -1840.912207, "press": 1247.825875, "evdwl": -637.1043152, "ecoul": 4020.548863,
        "elong": -3211.525725, "ebond": 129.3, "eangle": 586.4, "edihed": -2753.4, "ke": 2665.296821,
        "pxx": 1200.0, "pyy": 1250.0, "pzz": 1293.5}


def lammps_info(text=HELP_GPU, path="/opt/lammps/bin/lmp") -> bench.LammpsInfo:
    h = bench.parse_lmp_help(text)
    return bench.LammpsInfo(path, "PATH", h["version"], h["packages"], h["accelerators"], h["gpu_present"])


def md_record(**variants) -> dict:
    return {"bench_version": 1, "date": "2026-11-20T12:00:00+00:00", "lammps": "4 Jul 2026", "packages": ["GPU"],
            "system": {"size": "pilot", "atoms": 9780}, "variants": variants, "notes": [], "skipped": ""}


CPU_EWALD = {"ns_per_day": 6.5, "atoms": 9780, "threads": 6, "accelerator": "omp", "steps": 2000}
GPU_C16 = {"ns_per_day": 10.6, "atoms": 9780, "threads": 1, "accelerator": "gpu", "gpu_api": "OpenCL",
           "check": "mismatch", "check_detail": "pressure 4,512.0 vs CPU 312.0 atm"}


# ------------------------------------------------------------------------------------------- lmp -h and logs
def test_parse_lmp_help_finds_version_packages_and_accelerators():
    h = bench.parse_lmp_help(HELP_WHEEL)
    assert h["version"] == "22 Jul 2025 - Update 4"
    assert {"KSPACE", "MOLECULE", "OPENMP", "CG-DNA"} <= set(h["packages"]) and "GPU" not in h["packages"]
    assert "Atom" not in h["packages"] and h["accelerators"] == {"OPENMP": {"api": "OpenMP", "precision": "double"}}
    assert h["gpu_present"] is None
    g = lammps_info()
    assert g.version == "4 Jul 2026" and g.has("gpu") and g.gpu_present is True
    assert g.accelerators["GPU"] == {"api": "OpenCL", "precision": "mixed"} and g.kokkos_on_gpu
    assert not lammps_info(HELP_GPU.replace("HIP Serial", "OpenMP Serial")).kokkos_on_gpu
    assert bench.parse_lmp_help(HELP_GPU.replace(": yes", ": no"))["gpu_present"] is False
    assert bench.parse_lmp_help("no lammps here") == {"version": "", "packages": [], "accelerators": {},
                                                      "gpu_present": None}


def test_version_regex_is_the_one_chem_external_uses():
    import re

    from quantabench.chem import external

    spec = external.PROGRAMS["lammps"]
    assert spec.version_regex == bench.VERSION_RE and spec.version_args == ("-h",) and spec.engine
    assert re.search(spec.version_regex, HELP_WHEEL).group(1) == "22 Jul 2025 - Update 4"
    assert "lmp" in spec.executables


def test_parse_performance_and_tagged_lines():
    p = bench.parse_performance(LOG_TIMED)
    assert p["ns_per_day"] == 11.483 and p["hours_per_ns"] == 2.090 and p["timesteps_per_s"] == 132.904
    assert p["katom_steps_per_s"] == 398.711 and p["steps"] == 1990 and p["atoms"] == 3000
    assert p["loop_s"] == 14.9732 and p["procs"] == 2 and (p["mpi"], p["omp_threads"]) == (1, 2)
    # an older LAMMPS without katom-step/s; the last Performance line of several wins; Matom-step/s
    old = "Loop time of 2 on 1 procs for 100 steps with 10 atoms\nPerformance: 4.320 ns/day, 5.556 hours/ns, 50.000 timesteps/s\n"
    assert bench.parse_performance(old)["ns_per_day"] == 4.32 and "katom_steps_per_s" not in bench.parse_performance(old)
    two = old + "Loop time of 1 on 1 procs for 50 steps with 10 atoms\nPerformance: 9.000 ns/day, 2.667 hours/ns, 104.000 timesteps/s, 1.5 Matom-step/s\n"
    q = bench.parse_performance(two)
    assert q["ns_per_day"] == 9.0 and q["steps"] == 50 and q["katom_steps_per_s"] == 1500.0
    assert bench.parse_performance("no performance") is None
    gpu_init = "-------\nDevice 0: AMD Radeon RX 9060 XT, 32 CUs, 16 GB, 2.7 GHZ (Mixed Precision)\n-------\n"
    assert bench.DEVICE_RE.search(gpu_init).group(2) == "AMD Radeon RX 9060 XT, 32 CUs, 16 GB, 2.7 GHZ (Mixed Precision)"
    assert bench.parse_tagged(LOG_TIMED, "QB_END") == {"temp": 301.2, "press": -12.5}   # the echoed print is skipped
    assert bench.parse_tagged("QB_RUN0 pe -1.5 press nan ke x\n", "QB_RUN0")["pe"] == -1.5
    assert math.isnan(bench.parse_tagged("QB_RUN0 press nan\n", "QB_RUN0")["press"])
    assert bench.parse_tagged(LOG_TIMED, "QB_RUN0") is None
    assert bench.lammps_errors("ok\nERROR: Lost atoms: original 3000 current 2999 (x.cpp:1)\n") == [
        "ERROR: Lost atoms: original 3000 current 2999 (x.cpp:1)"]


def test_variants_parse_validate_and_plan():
    assert bench.parse_variant("GPU-cutoff16") == bench.Variant("gpu", "cutoff16")
    assert bench.parse_variant("cpu-ewald").name == "cpu-ewald"
    for bad in ("gpu", "tpu-ewald", "cpu-cutoff5", "cpu-pme"):
        with pytest.raises(ValueError):
            bench.parse_variant(bad)
    assert bench.parse_variants(None) is None and bench.parse_variants(" , ") is None
    assert bench.parse_variants("gpu-cutoff16,default") == ["gpu-cutoff16", "cpu-ewald", "cpu-cutoff12",
                                                            "cpu-cutoff16", "gpu-cutoff12"]
    # the same dispersion rules as md.liquid
    from quantabench.md import liquid

    for name in ("ewald", "cutoff12", "cutoff16", "cutoff18", "cutoff12.5"):
        assert bench.dispersion_cutoff(name) == liquid.parse_dispersion(name)
    # default plan: GPU variants only with the GPU package; ewald never on the GPU
    run, refused, notes = bench.plan_variants(lammps_info(HELP_WHEEL), None)
    assert [v.name for v in run] == ["cpu-ewald", "cpu-cutoff12", "cpu-cutoff16"] and not refused
    # on Windows the note adds where to get a LAMMPS with the GPU package
    assert len(notes) == 1 and notes[0].startswith("this LAMMPS has no GPU package")
    run, refused, notes = bench.plan_variants(lammps_info(), None)
    assert [v.name for v in run] == list(bench.DEFAULT_VARIANTS) and not notes
    run, refused, _ = bench.plan_variants(lammps_info(HELP_WHEEL), ["gpu-cutoff16", "gpu-ewald", "kokkos-cutoff16",
                                                                  "cpu-cutoff18"])
    assert [v.name for v in run] == ["cpu-cutoff18"]
    assert refused["gpu-cutoff16"].startswith("this LAMMPS has no GPU package")
    assert "ewald runs on the CPU" in refused["gpu-ewald"] and "KOKKOS" in refused["kokkos-cutoff16"]
    no_gpu = lammps_info(HELP_GPU.replace(": yes", ": no"))
    assert "no compatible GPU" in bench.can_run(no_gpu, bench.Variant("gpu", "cutoff16"))[1]
    bare = lammps_info(HELP_WHEEL.replace("KSPACE ", ""))
    assert "KSPACE" in bench.can_run(bare, bench.Variant("cpu", "ewald"))[1]


def test_accelerator_switches():
    gpu = lammps_info()
    assert bench.accelerator_args(gpu, "cpu", 6) == (["-sf", "omp", "-pk", "omp", "6"], 6, "omp")
    assert bench.accelerator_args(gpu, "gpu", 6) == (["-sf", "hybrid", "gpu", "omp", "-pk", "gpu", "1", "device_type",
                                                      "generic", "pair/only", "on", "-pk", "omp", "6"], 6, "gpu")
    cuda = lammps_info(HELP_GPU.replace("GPU package API: OpenCL", "GPU package API: CUDA"))
    assert "device_type" not in bench.accelerator_args(cuda, "gpu", 6)[0]
    assert bench.accelerator_args(gpu, "kokkos", 6) == (["-k", "on", "g", "1", "-sf", "kk"], 1, "kokkos-gpu")
    omp_kk = lammps_info(HELP_GPU.replace("HIP Serial", "OpenMP"))
    assert bench.accelerator_args(omp_kk, "kokkos", 4)[0] == ["-k", "on", "t", "4", "-sf", "kk"]
    plain = lammps_info(HELP_WHEEL.replace(" OPENMP ", " "))
    assert bench.accelerator_args(plain, "cpu", 6) == ([], 1, "none")
    assert bench.default_threads() >= 1


# ------------------------------------------------------------------------------------------- the check
def test_consistency_check_on_synthetic_numbers():
    status, detail = bench.check_consistency(RUN0, {**RUN0, "press": RUN0["press"] + 0.8, "pe": RUN0["pe"] + 0.01})
    assert status == "ok" and "diff +0.8" in detail
    # the workstation's GPU NPT symptom: a pressure thousands of atm off at the same configuration
    bad = {**RUN0, "press": 4512.0, "elong": RUN0["elong"] + 30.0, "pe": RUN0["pe"] + 30.0, "pzz": 4600.0}
    status, detail = bench.check_consistency(RUN0, bad)
    assert status == "mismatch" and "diff +3,264.2" in detail and "differing: elong" in detail and "pzz" in detail
    # energy alone: the tolerance grows with the size of the terms (0.5 + 1e-4 x ~9,900 = 1.5 kcal/mol)
    assert bench.check_consistency(RUN0, {**RUN0, "pe": RUN0["pe"] + 1.0})[0] == "ok"
    assert bench.check_consistency(RUN0, {**RUN0, "pe": RUN0["pe"] + 3.0})[0] == "mismatch"
    assert bench.check_consistency(RUN0, {**RUN0, "press": RUN0["press"] + 25.0})[0] == "mismatch"
    assert bench.check_consistency(RUN0, {**RUN0, "press": RUN0["press"] + 25.0}, pressure_tol=30)[0] == "ok"
    assert bench.check_consistency(RUN0, {**RUN0, "press": float("nan")})[0] == "mismatch"
    assert bench.check_consistency(None, RUN0) == ("skipped", "no CPU reference at run 0")
    assert bench.check_consistency(RUN0, {"pe": 1.0})[0] == "skipped"
    assert bench.verified({"check": "ok"}) and bench.verified({}) and not bench.verified({"check": "mismatch"})
    assert not bench.verified({"check": "skipped"})


# ------------------------------------------------------------------------------------------- the registry
def test_registry_round_trip_with_and_without_md_throughput():
    md = md_record(**{"cpu-ewald": CPU_EWALD, "gpu-cutoff16": GPU_C16})
    p = MachineProfile(name="bench-pc", throughput={"mmff@cpu": 9000.0}, md_throughput=md)
    q = MachineProfile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q == p and q.md_throughput["variants"]["gpu-cutoff16"]["check"] == "mismatch"
    # a registry file from before MD-BENCH-1 has no key: it loads, with {}
    old = p.to_dict()
    del old["md_throughput"]
    assert MachineProfile.from_dict(old).md_throughput == {}
    profile.registry_dir().mkdir(parents=True, exist_ok=True)
    (profile.registry_dir() / "bench-pc.json").write_text(json.dumps(old), encoding="utf-8")
    loaded = profile.load_profile("bench-pc")
    assert loaded.md_throughput == {} and loaded.throughput == {"mmff@cpu": 9000.0}
    assert "md:" not in "\n".join(profile.describe(loaded))
    # wrong types fall back or are dropped, never hide the machine
    bad = {**old, "md_throughput": {"variants": {"cpu-ewald": {"ns_per_day": "fast", "atoms": 9780},
                                                 "gpu-cutoff16": {"ns_per_day": -1}, "junk": 5,
                                                 "cpu-cutoff12": {"ns_per_day": float("inf")}}, "lammps": "x"}}
    r = MachineProfile.from_dict(bad).md_throughput
    assert r["lammps"] == "x" and set(r["variants"]) == {"cpu-ewald", "gpu-cutoff16", "cpu-cutoff12"}
    assert not bench.rated(r)
    assert MachineProfile.from_dict({**old, "md_throughput": [1, 2]}).md_throughput == {}
    assert MachineProfile.from_dict({**old, "md_throughput": {"variants": 3}}).md_throughput == {"variants": {}}


def test_merge_keeps_variants_that_were_not_measured_again():
    prev = md_record(**{"cpu-ewald": CPU_EWALD, "gpu-cutoff16": GPU_C16})
    new = md_record(**{"gpu-cutoff16": {**GPU_C16, "ns_per_day": 11.0, "check": "ok"}})
    m = bench.merge(prev, new)
    assert m["variants"]["cpu-ewald"] == CPU_EWALD and m["variants"]["gpu-cutoff16"]["check"] == "ok"
    failed = md_record(**{"cpu-ewald": {"error": "ERROR: Lost atoms"}})
    assert "ns_per_day" not in bench.merge(prev, failed)["variants"]["cpu-ewald"]       # a failure replaces
    missing = {"date": "2026-11-21", "variants": {}, "skipped": "LAMMPS not found (...)"}
    kept = bench.merge(prev, missing)
    assert kept["variants"] == prev["variants"] and kept["last_attempt"]["skipped"].startswith("LAMMPS not found")
    assert bench.merge(None, missing) == missing and bench.merge(prev, {}) == prev


def test_describe_shows_md_line_and_failures():
    md = md_record(**{"cpu-ewald": CPU_EWALD, "gpu-cutoff16": GPU_C16,
                      "gpu-cutoff12": {"error": "ERROR: Could not find/initialize a specified accelerator device"}})
    lines = bench.describe_lines(md)
    assert lines[0] == ("  md:         cpu-ewald 6.5 ns/day (9,780 atoms, 6 thr), gpu-cutoff16 10.6 ns/day "
                        "(9,780 atoms, OpenCL) [pressure check: mismatch] (LAMMPS 4 Jul 2026)")
    assert lines[1].startswith("  md benchmark gpu-cutoff12 failed: ERROR: Could not find")
    assert bench.short_text(md) == "cpu-ewald 6.5, gpu-cutoff16 10.6 (unverified)"
    assert bench.describe_lines({"skipped": "LAMMPS not found (x)", "variants": {}}) == [
        "  md:         not measured (LAMMPS not found (x))"]
    assert bench.describe_lines({}) == [] and bench.short_text({}) == ""
    same = md_record(**{"cpu-ewald": CPU_EWALD, "cpu-cutoff12": {**CPU_EWALD, "ns_per_day": 123.4}})
    assert bench.describe_lines(same)[0].startswith("  md:         cpu-ewald 6.5 ns/day (9,780 atoms, 6 thr), "
                                                    "cpu-cutoff12 123 ns/day (LAMMPS")
    p = MachineProfile(name="bench-pc", md_throughput=md)
    text = "\n".join(profile.describe(p))
    assert "md:         cpu-ewald 6.5 ns/day" in text and "throughput: not measured" in text


# ------------------------------------------------------------------------------------------- estimates
def test_estimate_scales_with_atoms_and_leaves_unverified_out():
    md = md_record(**{"cpu-ewald": CPU_EWALD, "cpu-cutoff16": {**CPU_EWALD, "ns_per_day": 8.0},
                      "gpu-cutoff16": GPU_C16})
    e = bench.best_estimate(md, 9780, 13.0, dispersion="ewald")
    assert e.variant == "cpu-ewald" and math.isclose(e.hours, 13 / 6.5 * 24) and e.verified     # 48 h
    e = bench.best_estimate(md, 2 * 9780, 13.0, dispersion="ewald")
    assert math.isclose(e.ns_per_day, 3.25) and math.isclose(e.hours, 96.0)                  # linear in atoms
    e = bench.best_estimate(md, 9780, 13.0, dispersion="cutoff16")
    assert e.variant == "cpu-cutoff16"                                       # the GPU failed its pressure check
    e = bench.best_estimate(md, 9780, 13.0, dispersion="cutoff16", include_unverified=True)
    assert e.variant == "gpu-cutoff16" and not e.verified
    e = bench.best_estimate(md, 9780, 13.0, dispersion="cutoff18")               # scaled from the nearest cutoff
    assert e.variant == "cpu-cutoff18" and math.isclose(e.ns_per_day, 8.0 * (16 / 18) ** 3) and "16/18" in e.note
    assert bench.best_estimate(md, 9780, 13.0).variant == "cpu-cutoff16"         # any dispersion: the fastest
    assert [x.variant for x in bench.estimate_hours(md, 9780, 1.0)] == ["cpu-cutoff16", "cpu-ewald"]
    assert bench.estimate_hours(md, 9780, 1.0, accelerator="gpu") == []
    assert bench.best_estimate({}, 9780, 1.0) is None
    with pytest.raises(ValueError):
        bench.estimate_hours(md, 0, 1.0)
    assert bench.format_hours(0.5) == "30 min" and bench.format_hours(12.34) == "12.3 h"
    assert bench.format_hours(96) == "4.0 days (96 h)"
    lines = bench.estimate_lines(9780, 13.0, "cutoff16", profiles=[
        MachineProfile(name="bench-pc", md_throughput=md), MachineProfile(name="bench-mac")], this="bench-pc")
    assert len(lines) == 1 and lines[0].startswith("bench-pc (this machine): 39.0 h with cpu-cutoff16")
    assert "gpu-cutoff16 would take 29.4 h but is unverified (pressure check: mismatch)" in lines[0]


def test_md_estimate_cli_and_the_liquid_slab_hook(tmp_path, capsys):
    md = md_record(**{"cpu-ewald": CPU_EWALD, "gpu-cutoff16": GPU_C16})
    profile.save_profile(MachineProfile(name="bench-pc", md_throughput=md, updated="2026-11-20T12:00:00+00:00"))
    assert cli.main(["md", "estimate", "--atoms", "9780", "--ns", "13", "--dispersion", "ewald"]) == 0
    out = capsys.readouterr().out
    assert "13 ns of 9,780 atoms (ewald, slab)" in out and "bench-pc" in out and "cpu-ewald" in out
    assert "2.0 days (48 h)" in out and "bulk box, from the bulk benchmark" in out      # no slab record: noted
    assert cli.main(["md", "estimate", "--atoms", "9780", "--ns", "13", "--include-unverified", "--all-variants",
                     "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["estimates"]
    assert [r["variant"] for r in rows] == ["gpu-cutoff16", "cpu-ewald"] and rows[0]["check"] == "mismatch"
    assert cli.main(["md", "estimate", "--ns", "13"]) == 2
    assert cli.main(["md", "estimate", "--atoms", "10", "--ns", "1", "--dispersion", "pme"]) == 2
    # md liquid-slab prints the estimate for its own atoms and ns and adds it to README.txt
    out_dir = tmp_path / "c6f14"
    assert cli.main(["md", "liquid-slab", "--smiles", "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "--name",
                     "c6f14", "-o", str(out_dir), "--packing", "lattice", "--lateral", "26", "--thickness", "20",
                     "--bulk-ns", "1", "--settle-ns", "2", "--production-ns", "10", "--engine", "lammps"]) == 0
    out = capsys.readouterr().out
    n_atoms = json.loads((out_dir / "liquid_setup.json").read_text(encoding="utf-8"))["n_atoms"]
    assert f"Run time, 13 ns of {n_atoms:,} atoms (ewald)" in out and "bench-pc" in out
    readme = (out_dir / "README.txt").read_text(encoding="utf-8")
    assert "RUN TIME ESTIMATE" in readme and "with cpu-ewald" in readme
    assert cli.main(["md", "estimate", str(out_dir)]) == 0
    assert f"13 ns of {n_atoms:,} atoms (ewald, slab)" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- the wiring
def test_bench_lists_take_md_without_changing_the_default():
    engines = {"mmff": True, "gfn1-xtb": True, "gfn2-xtb": False, "aimnet2": True, "xtb-gfn2": True}
    assert profile.split_md(None) == (None, False) and profile.split_md(["md"]) == ([], True)
    assert profile.split_md(["default", " md"]) == (None, True) and profile.split_md(["mmff", "md"]) == (["mmff"], True)
    assert profile.expand_methods(None, engines) == ["mmff", "gfn1-xtb", "aimnet2"]
    assert profile.expand_methods(["xtb-gfn2", "default"], engines) == ["xtb-gfn2", "mmff", "gfn1-xtb", "aimnet2"]
    assert profile._parse_methods("default", engines) == ["mmff", "gfn1-xtb", "aimnet2"]          # unchanged
    assert profile._parse_methods("md", engines) == ["md"]
    assert profile._parse_methods("default,md", engines) == ["mmff", "gfn1-xtb", "aimnet2", "md"]
    assert profile._parse_methods("mmff,md", engines) == ["mmff", "md"]
    with pytest.raises(ValueError, match="default, md"):
        profile._parse_methods("mmff,lammps", engines)


def test_local_profile_runs_md_only_when_asked_and_merges(monkeypatch):
    calls = {"engines": [], "md": []}

    def fake_benchmark(methods, devices=None, details=None, **k):
        calls["engines"].append(list(methods))
        return {f"{m}@cpu": 100.0 for m in methods}

    def fake_md(variants=None, **kw):
        calls["md"].append((variants, kw.get("size"), kw.get("threads")))
        return md_record(**{"gpu-cutoff16": {**GPU_C16, "check": "ok"}})

    monkeypatch.setattr(profile, "benchmark", fake_benchmark)
    monkeypatch.setattr(bench, "run_md_benchmark", fake_md)
    me = profile.machine_name()
    profile.save_profile(MachineProfile(name=me, throughput={"mmff@cpu": 1.0},
                                        md_throughput=md_record(**{"cpu-ewald": CPU_EWALD})))
    p = profile.local_profile(bench=True, methods=["mmff"])
    assert calls == {"engines": [["mmff"]], "md": []} and p.md_throughput["variants"] == {"cpu-ewald": CPU_EWALD}
    p = profile.local_profile(bench=True, methods=["md"], md_options={"size": "pilot", "variants": ["gpu-cutoff16"],
                                                                      "threads": 6})
    assert calls["engines"] == [["mmff"]] and calls["md"] == [(["gpu-cutoff16"], "pilot", 6)]     # no engine run
    assert p.throughput == {"mmff@cpu": 1.0} and set(p.md_throughput["variants"]) == {"cpu-ewald", "gpu-cutoff16"}
    profile.local_profile(bench=True)                                          # the default: engines, no MD
    assert len(calls["md"]) == 1


def test_machines_cli_bench_md(capsys, monkeypatch):
    seen = {}

    def fake_md(variants=None, **kw):
        seen.update(variants=variants, **kw)
        return md_record(**{"cpu-ewald": {**CPU_EWALD, "atoms": 3000, "threads": 4}})

    monkeypatch.setattr(bench, "run_md_benchmark", fake_md)
    monkeypatch.setattr(profile, "benchmark", lambda *a, **k: pytest.fail("no engine benchmark for --bench md"))
    assert cli.main(["machines", "--bench", "md", "--md-size", "pilot", "--md-variants", "cpu-ewald,gpu-cutoff16",
                     "--md-threads", "6", "--max-seconds", "45"]) == 0
    out = capsys.readouterr().out
    assert seen["variants"] == ["cpu-ewald", "gpu-cutoff16"] and seen["size"] == "pilot" and seen["threads"] == 6
    assert seen["max_seconds"] == 45 and "MD benchmark on" in out and "Benchmarking" not in out
    assert "md:         cpu-ewald 6.5 ns/day (3,000 atoms, 4 thr)" in out
    assert profile.load_profile(profile.machine_name()).md_throughput["variants"]["cpu-ewald"]["threads"] == 4
    assert cli.main(["machines", "--bench", "md", "--md-variants", "tpu-ewald"]) == 2
    assert "tpu-ewald" in capsys.readouterr().err


def test_worker_benchmarks_md_only_when_listed():
    from quantabench.dist.worker import Worker, md_measured

    def worker(bench_mode, methods, prof):
        w = SimpleNamespace(bench=bench_mode, bench_methods=methods, profile=prof)
        w._methods_to_measure = lambda: Worker._methods_to_measure(w)
        return w

    empty, measured = MachineProfile(name="w"), MachineProfile(name="w", throughput={"mmff@cpu": 5.0})
    with_md = MachineProfile(name="w", throughput={"mmff@cpu": 5.0}, md_throughput=md_record(**{"cpu-ewald": CPU_EWALD}))
    assert md_measured(with_md) and not md_measured(measured)
    cases = [("auto", None, empty, None), ("auto", None, measured, []),          # None: the default engines
             ("auto", ["default", "md"], measured, ["md"]),            # only what is missing: no engines again
             ("auto", ["default", "md"], empty, ["default", "md"]), ("auto", ["md"], empty, ["md"]),
             ("auto", ["md"], with_md, []), ("auto", ["mmff", "md"], with_md, []),
             ("always", ["default", "md"], with_md, ["default", "md"]), ("always", None, with_md, None)]
    for mode, methods, prof, want in cases:
        w = worker(mode, methods, prof)
        assert Worker._methods_to_measure(w) == want, (mode, methods, prof)
        assert Worker.needs_benchmark(w) == (mode == "always" or want != [])
    assert not Worker.needs_benchmark(worker("never", ["md"], empty))


def test_missing_lammps_is_skipped_with_a_reason(monkeypatch):
    from quantabench.chem import external

    monkeypatch.setattr(external, "find_program", lambda *a, **k: None)
    rec = bench.run_md_benchmark()
    assert rec["skipped"].startswith("LAMMPS not found (") and rec["variants"] == {}
    assert bench.describe_lines(rec) == [f"  md:         not measured ({rec['skipped']})"]
    me = profile.machine_name()
    profile.save_profile(MachineProfile(name=me, md_throughput=md_record(**{"cpu-ewald": CPU_EWALD})))
    p = profile.local_profile(bench=True, methods=["md"])
    assert p.md_throughput["variants"] == {"cpu-ewald": CPU_EWALD}                  # the old numbers are kept
    assert p.md_throughput["last_attempt"]["skipped"].startswith("LAMMPS not found")
    wsl = external.ProgramInfo("lammps", "wsl:/usr/bin/lmp", "settings")
    monkeypatch.setattr(external, "find_program", lambda *a, **k: wsl)
    assert "through WSL" in bench.find_lammps()[1]


# ------------------------------------------------------------------------------------------- real LAMMPS
def _real_lammps():
    info, why = bench.find_lammps(refresh=True)
    if info is None:
        pytest.skip(why)
    return info


def test_find_lammps_reads_lmp_h():
    info = _real_lammps()
    assert info.version and info.has("KSPACE") and info.has("MOLECULE") and info.path


def test_real_short_benchmark_and_the_check(monkeypatch):
    """A tiny box (800 atoms), 200 steps per variant: the inputs run, ns/day is parsed, and the CPU/GPU check passes
    for a variant with the same physics and fails for one without the bulk's tail correction (a stand-in for a GPU
    package that gets the pressure wrong: the GPU package itself is not in the PyPI wheel)."""
    info = _real_lammps()
    monkeypatch.setitem(bench.SYSTEMS, "tiny", {"config": {"lateral": 26.0, "n_molecules": 40}})
    fake_gpu = bench.LammpsInfo(info.path, info.source, info.version, sorted({*info.packages, "GPU"}),
                                {**info.accelerators, "GPU": {"api": "OpenCL", "precision": "mixed"}}, True)
    real_args, real_input = bench.accelerator_args, bench.bench_input

    def args(lmp, accelerator, threads):        # the "GPU" runs on the CPU here
        return real_args(lmp, "cpu", 1) if accelerator == "gpu" else real_args(lmp, accelerator, threads)

    def inp(tm, dispersion, *, label="", **kw):  # ... and gpu-cutoff16 gets the pressure wrong
        text = real_input(tm, dispersion, label=label, **kw)
        return text.replace("tail yes", "tail no") if label == "gpu-cutoff16" else text

    monkeypatch.setattr(bench, "accelerator_args", args)
    monkeypatch.setattr(bench, "bench_input", inp)
    logs = []
    rec = bench.run_md_benchmark(["cpu-ewald", "gpu-cutoff12", "gpu-cutoff16"], size="tiny", threads=1,
                                 max_seconds=20, steps=200, lammps=fake_gpu, log=logs.append)
    v = rec["variants"]
    assert rec["system"]["atoms"] == 800 and rec["lammps"] == info.version and not rec["skipped"], logs
    for name in ("cpu-ewald", "gpu-cutoff12", "gpu-cutoff16"):
        assert not v[name]["error"], (name, v[name])
        assert v[name]["ns_per_day"] > 0 and v[name]["atoms"] == 800 and v[name]["steps"] == 200
        assert set(bench.RUN0_KEYS) <= set(v[name]["run0"])
    assert "check" not in v["cpu-ewald"]
    assert "mesh spacing ≤ 2.25 Å" in v["cpu-ewald"]["long_range"]          # MD-LR: the settings it timed
    assert v["gpu-cutoff16"]["long_range"].startswith("16 Å LJ cutoff + tail")
    assert v["gpu-cutoff12"]["check"] == "ok", v["gpu-cutoff12"]["check_detail"]
    assert v["gpu-cutoff16"]["check"] == "mismatch" and "pressure" in v["gpu-cutoff16"]["check_detail"]
    assert v["gpu-cutoff12"]["reference_run0"]["press"] == pytest.approx(v["gpu-cutoff12"]["run0"]["press"], abs=1e-6)
    assert any("check mismatch" in line for line in logs)
    # the box is cached: a second run builds nothing
    logs.clear()
    again = bench.run_md_benchmark(["cpu-cutoff12"], size="tiny", threads=1, max_seconds=10, steps=50,
                                   lammps=fake_gpu, log=logs.append)
    assert again["variants"]["cpu-cutoff12"]["ns_per_day"] > 0 and not any("building" in x for x in logs)
    # and it lands in the profile, readable by the estimate
    p = MachineProfile(name="here", md_throughput=bench.merge(rec, again))
    assert bench.best_estimate(p.md_throughput, 8000, 1.0, dispersion="cutoff16").variant == "cpu-cutoff16"
