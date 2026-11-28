"""SCHED-3 PR 1: the device planner (``dist.scheduler``). Synthetic machine profiles shaped like the data of Brandon's two
computers (not their measurements): a 12-core / 24-thread workstation with a Radeon whose LAMMPS GPU package runs the
pair style beside 4-12 PPPM threads and whose OpenMM runs on one thread, and a 14-core Mac whose Apple GPU is slower
than its CPU for AIMNet2 and runs OpenMM in single precision. The choice per job, the lean assignment, the shared-core
model, side by side against one after the other, learned rates, the job specs, MD folders and the CLI."""

from __future__ import annotations

import json

import pytest

from quantabench import cli
from quantabench.dist import profile as P
from quantabench.dist import scheduler as S
from quantabench.dist import threads as T


def tower(**kw) -> P.MachineProfile:
    """A workstation like the Windows one: 12 cores / 24 threads, a GPU, LAMMPS with the GPU package, OpenMM."""
    base = dict(
        name="tower-test", os="Windows 11", physical_cores=12, logical_cores=24, ram_gb=32.0,
        gpus=[{"index": 0, "name": "Radeon (test)", "memory_gb": 16.0, "backend": "rocm"}],
        throughput={"aimnet2@cpu": 3600.0, "aimnet2@cuda:0": 36000.0, "gfn2-xtb@cpu": 300.0, "mmff@cpu": 18000.0},
        benchmarks={k: {"threads": 23, "set": "mixed-10"} for k in ("aimnet2@cpu", "aimnet2@cuda:0", "gfn2-xtb@cpu",
                                                                    "mmff@cpu")},
        md_throughput={"bench_version": 4, "variants": {
            "cpu-cutoff16@slab": {"ns_per_day": 14.0, "atoms": 3800, "threads": 12, "box_A": [40, 40, 78],
                                  "breakdown": {"Pair": 58.0, "Kspace": 28.0}},
            "gpu-cutoff16@slab": {"ns_per_day": 55.0, "atoms": 3800, "threads": 12, "check": "ok", "box_A": [40, 40, 78],
                                  "breakdown": {"Pair": 8.0, "Kspace": 48.0, "Bond": 27.0}},
            "cpu-ewald@slab": {"ns_per_day": 9.0, "atoms": 3800, "threads": 12, "box_A": [40, 40, 78],
                               "long_range": "ewald"}}})
    base.update(kw)
    return P.MachineProfile(**base)


def with_openmm(p: P.MachineProfile, ns_per_day: float = 330.0, precision: str = "mixed") -> P.MachineProfile:
    p.md_throughput["variants"]["openmm-gpu-cutoff16@slab"] = {
        "ns_per_day": ns_per_day, "atoms": 3800, "threads": 1, "check": "ok", "precision": precision}
    return p


def mac() -> P.MachineProfile:
    """A Mac like the M3 Max: 14 cores, no SMT, AIMNet2 slower on MPS than on the CPU, OpenMM single precision."""
    return P.MachineProfile(
        name="mac-test", os="Darwin", physical_cores=14, logical_cores=14, ram_gb=36.0,
        gpus=[{"index": 0, "name": "Apple GPU", "memory_gb": 36.0, "backend": "mps"}],
        throughput={"aimnet2@cpu": 7104.0, "aimnet2@mps": 5179.0},
        benchmarks={"aimnet2@cpu": {"threads": 13, "set": "mixed-10"}, "aimnet2@mps": {"threads": 13, "set": "mixed-10"}},
        md_throughput={"bench_version": 4, "variants": {
            "cpu-cutoff16@slab": {"ns_per_day": 24.2, "atoms": 3800, "threads": 9},
            "openmm-gpu-cutoff16@slab": {"ns_per_day": 189.0, "atoms": 3800, "threads": 1, "check": "ok",
                                         "precision": "single", "platform": "OpenCL"}}})


CREST_FIT = T.CostModel(power=3.9, speedup_power=0.5, scale_s=5e-4, source="fit tower-test gfnff quick (test)",
                        scatter=2.8)


def view(p, threads=23, **kw) -> S.Machine:
    return S.machine_view(p, threads=threads, cost_models=kw.pop("cost_models", {"gfnff quick": CREST_FIT}), **kw)


def md(**kw) -> S.PlanJob:
    base = dict(atoms=9780, ns=13.0, dispersion="cutoff16", engine="lammps", name="slab")
    base.update(kw)
    return S.PlanJob("md", **base)


# ----------------------------------------------------------------------------------------------- one job
def test_lammps_gpu_md_needs_threads_and_has_a_lean_assignment():
    m = view(tower())
    plan = S.plan_job(md(), m)
    cpu, gpu_only, both = plan.options
    assert plan.best is both and both.gpu and both.threads == 12          # at most the physical cores
    assert both.variant == "gpu-cutoff16" and both.source == "benchmark"
    assert gpu_only.threads == 1 and gpu_only.hours > 1.8 * both.hours    # PPPM on one thread: much slower
    assert cpu.hours > 3 * both.hours and not cpu.gpu
    lean = plan.lean
    assert lean is not None and lean.gpu and 4 <= lean.threads <= 8      # "the GPU + 4-6 threads" of the pilots
    assert lean.hours <= both.hours * (1 + S.LEAN)
    text = " ".join(plan.why)
    assert "PPPM" in text and "leaves" in text and "beside another job" in text
    # the atoms scale the benchmark: 3,800 -> 9,780 atoms is 2.57x the time per ns (no box given: the atom rule)
    assert both.rate == pytest.approx(55.0 * 3800 / 9780, rel=1e-6)
    assert both.hours == pytest.approx(13.0 / both.rate * 24.0)


def test_openmm_on_the_gpu_takes_one_thread_and_the_engine_is_named():
    m = view(with_openmm(tower()))
    plan = S.plan_job(md(engine=""), m)
    assert plan.best.gpu and plan.best.threads == 1 and plan.best.variant == "openmm-gpu-cutoff16"
    assert len(plan.options) == 2         # GPU + N threads is the same as GPU only for OpenMM: not listed twice
    text = " ".join(plan.why)
    assert "one CPU thread" in text and "--engine openmm" in text
    # a folder written for LAMMPS keeps LAMMPS' variants
    assert S.plan_job(md(engine="lammps"), m).best.variant == "gpu-cutoff16"


def test_the_macs_gpu_is_single_precision_and_its_aimnet2_stays_on_the_cpu():
    m = view(mac(), threads=13)
    plan = S.plan_job(md(engine=""), m)
    assert plan.best.variant == "openmm-gpu-cutoff16"
    assert any("single precision" in n for n in plan.best.notes)
    batch = S.plan_job(S.PlanJob("build", method="aimnet2", molecules=5000), m)
    assert not batch.best.gpu                                # MPS 5,179/h vs CPU 7,104/h per process
    assert "slower" in " ".join(batch.why)
    gpu_only = batch.options[1]
    assert gpu_only.known and gpu_only.hours > batch.best.hours


def test_ewald_dispersion_has_no_lammps_gpu_style():
    plan = S.plan_job(md(dispersion="ewald"), view(tower()))
    assert not plan.best.gpu and plan.best.variant == "cpu-ewald"
    gpu = plan.options[1]
    assert not gpu.possible and "Ewald" in gpu.why_not


def test_a_machine_without_an_md_benchmark_has_no_md_prediction():
    plan = S.plan_job(md(), view(tower(md_throughput={})))
    assert plan.best is None and "machines --bench md" in plan.why[0]


def test_aimnet2_with_a_fast_gpu_uses_the_gpu_and_cpu_lanes():
    m = view(tower())
    plan = S.plan_job(S.PlanJob("build", method="aimnet2", molecules=5000), m)
    cpu, gpu_only, both = plan.options
    assert plan.best is both and both.lanes.startswith("1 GPU lane x 4 + ")
    assert gpu_only.hours > both.hours > 0 and cpu.hours > gpu_only.hours
    # CPU lanes of a PyTorch engine: 2-thread lanes on the physical cores (dist.lanes.plan_auto), so 12 of 23
    assert cpu.threads == 12 and cpu.lanes == "6 CPU lanes x 2"
    # the measured ratio of 12 x 1 MMFF lanes to one 12-thread process (3.6x on the workstation): q = 0.48
    mm = S.plan_job(S.PlanJob("build", method="mmff", molecules=1000), view(tower(), threads=12))
    assert mm.best.lanes == "12 CPU lanes x 1"
    assert mm.best.rate == pytest.approx(18000.0 * 12 * (1 / 12) ** S.THREAD_POWER["mmff"], rel=1e-6)
    assert any("short protocol" in n for n in mm.best.notes)


def test_aimnet2_with_a_slow_gpu_stays_on_the_cpu():
    m = view(tower(throughput={"aimnet2@cpu": 3600.0, "aimnet2@cuda:0": 1200.0}))
    plan = S.plan_job(S.PlanJob("build", method="aimnet2", molecules=5000), m)
    assert not plan.best.gpu and plan.best.lanes == "6 CPU lanes x 2"


def test_dxtb_stays_on_the_cpu_and_never_mixes_precisions():
    m = view(tower())
    plan = S.plan_job(S.PlanJob("build", method="gfn2-xtb", molecules=500), m)
    assert not plan.best.gpu
    gpu = plan.options[1]
    assert not gpu.possible and "dxtb" in gpu.why_not and "5-8x" in gpu.why_not
    # with a measured GPU rate (dxtb GPU = on): one lane on the GPU, no CPU lanes beside it
    m2 = view(tower(throughput={"gfn2-xtb@cpu": 300.0, "gfn2-xtb@cuda:0": 60.0}))
    p2 = S.plan_job(S.PlanJob("build", method="gfn2-xtb", molecules=500), m2)
    assert not p2.best.gpu
    assert all(x.threads <= 4 for x in p2.options if x.gpu and x.known)


def test_cpu_only_engines_have_no_gpu_option():
    plan = S.plan_job(S.PlanJob("build", method="mmff", molecules=100), view(tower()))
    assert not plan.options[1].possible and "CPU only" in plan.options[1].why_not


def test_crest_is_cpu_only_and_timed_by_the_fitted_cost_model():
    m = view(tower())
    plan = S.plan_job(S.PlanJob("crest", molecules=1000, size=40.0, threads=10), m)
    best = plan.best
    assert not best.gpu and best.threads == 10 and best.lanes == "2 CPU lanes x 5"
    per = CREST_FIT.seconds(40.0, 5)
    assert best.hours == pytest.approx(1000 * per / 2 / 3600.0, rel=1e-9)      # 10 threads on 12 cores: no SMT
    assert not plan.options[1].possible and "WSL" in plan.options[1].why_not
    # all 23 threads: 3 x 5 + 2 x 4 lanes on 12 cores, each thread ~58 % of a core
    full = S.plan_job(S.PlanJob("crest", molecules=1000, size=40.0), m).best
    assert full.lanes == "3 CPU lanes x 5 + 2 CPU lanes x 4" and any("SMT" in n for n in full.notes)
    assert full.hours < best.hours
    # no fit on that machine: the time is unknown and the planner says how to get one
    none = S.plan_job(S.PlanJob("crest", molecules=1000, size=40.0), view(tower(), cost_models={}))
    assert none.best is None and "timings --fit" in none.why[0]


def test_capacity_counts_smt_threads_as_a_fraction_of_a_core():
    assert S.capacity(12, 12) == 12 and S.capacity(24, 12) == pytest.approx(12 + 12 * S.SMT_GAIN)
    assert S.thread_speed(6, 12) == 1.0 and S.thread_speed(24, 12) == pytest.approx((12 + 12 * S.SMT_GAIN) / 24)


def test_a_fixed_device_or_thread_count_is_kept():
    m = view(tower())
    plan = S.plan_job(S.PlanJob("build", method="aimnet2", molecules=5000, device="cpu"), m)
    assert not plan.best.gpu and not plan.options[1].possible and "CPU" in plan.options[1].why_not
    plan = S.plan_job(md(threads=6), m)
    assert plan.best.threads == 6


# ----------------------------------------------------------------------------------------------- learned rates
def test_this_runs_segments_replace_the_benchmark():
    m = view(tower())
    segs = [{"n": 1, "part": "bulk", "machine": "tower-test", "variant": "cpu-cutoff16", "threads": 8,
             "ns_per_day": 9.0},
            {"n": 2, "part": "slab", "machine": "tower-test", "variant": "gpu-cutoff16", "threads": 6,
             "ns_per_day": 22.1},
            {"n": 3, "part": "slab", "machine": "other-box", "variant": "gpu-cutoff16", "threads": 6,
             "ns_per_day": 99.0}]
    plan = S.plan_job(md(part="slab", segments=segs), m)
    best = plan.best
    assert best.source == "this run's segments" and "segment 2" in best.detail
    # its own 22.1 ns/day on 6 threads, scaled to 12 by the GPU-wait share of the benchmark and Amdahl's law
    g, a = 0.08, S._amdahl
    assert best.rate == pytest.approx(22.1 / (g + (1 - g) * a(12) / a(6)), rel=1e-9)
    # the CPU segment ran another part (the bulk box): the CPU stays on the slab benchmark
    assert plan.options[0].source == "benchmark"


def test_learned_build_rates_from_run_timings():
    rows = []
    for i in range(5):
        for stage in ("prepare", "conformers", "refine"):
            rows.append({"run_id": "r1", "mol_id": f"m{i}", "attempt": 1, "stage": stage, "engine": "rdkit",
                         "attempt_outcome": "ok", "attempt_wall_s": 20.0 + i, "size": 20.0, "mol_method": "aimnet2",
                         "solvent": "", "machine": "tower-test", "lane_device": "cpu", "lane_threads": 2,
                         "device": "cpu"})
    rows.append({"run_id": "r1", "mol_id": "c", "attempt": 1, "stage": "conformers", "engine": "crest 3.0.2",
                 "attempt_outcome": "ok", "attempt_wall_s": 900.0, "size": 20.0, "mol_method": "aimnet2",
                 "machine": "tower-test", "lane_threads": 2})
    learned = S.learn_build_rates(rows)
    got = learned[("tower-test", "aimnet2", "cpu")]
    assert got["molecules"] == 5 and got["s_per_atom"] == pytest.approx(22.0 / 20.0) and got["threads"] == 2
    m = view(tower(), learned=learned)
    cpu = S.plan_job(S.PlanJob("build", method="aimnet2", molecules=600, size=20.0), m).options[0]
    assert cpu.source == "run timings" and cpu.rate == pytest.approx(6 * 3600.0 / 22.0, rel=1e-9)


# ----------------------------------------------------------------------------------------------- two jobs
def test_openmm_md_and_a_crest_batch_run_side_by_side():
    m = view(with_openmm(tower()))
    x = S.side_by_side(md(engine="", name="slab"), S.PlanJob("crest", "crest", molecules=1000, size=40.0), m)
    assert x.verdict == "side by side"
    slab, crest = x.side
    assert slab.gpu and slab.threads == 1 and not crest.gpu and 16 <= crest.threads <= 22   # SMT: fewer may be faster
    assert x.side_hours < x.serial_hours
    text = " ".join(x.lines)
    assert "slab about as soon" in text and "crest done" in text and "sooner" in text


def test_two_cpu_batches_share_the_cores_and_run_one_after_the_other():
    m = view(tower(throughput={"aimnet2@cpu": 3600.0, "gfn2-xtb@cpu": 300.0}))
    x = S.side_by_side(S.PlanJob("build", "a", method="aimnet2", molecules=3000),
                       S.PlanJob("build", "b", method="gfn2-xtb", molecules=300), m)
    assert x.verdict == "one after the other"
    # with the cores shared they cannot beat one after the other by the margin
    assert x.side_hours > (1 - S.SIDE_MARGIN) * x.serial_hours


def test_a_running_job_comes_first_one_after_the_other():
    m = view(tower())
    crest = S.PlanJob("crest", "crest water", molecules=1000, size=40.0, threads=10, running=True)
    batch = S.PlanJob("build", "aimnet2", method="aimnet2", molecules=5000)
    x = S.side_by_side(crest, batch, m)
    assert x.serial_order == ["crest water", "aimnet2"]       # it cannot wait for the short one
    assert x.verdict == "side by side"
    assert x.side[0].threads == 10 and x.side[1].gpu
    assert "it runs already" in x.lines[0]
    # not running: the short job would go first, and waiting for it costs the long one little
    y = S.side_by_side(S.PlanJob("crest", "crest", molecules=1000, size=40.0, threads=10), batch, m)
    assert y.serial_order == ["aimnet2", "crest"]


def test_gpu_md_beside_a_cpu_batch_says_what_finishes_later():
    m = view(tower())
    x = S.side_by_side(md(), S.PlanJob("crest", "crest", molecules=1000, size=40.0, threads=10), m)
    assert x.verdict == "side by side"
    text = " ".join(x.lines)
    assert "slab" in text and "later (after" in text      # the MD job gets fewer threads beside CREST
    assert "crest done" in text


def test_a_slowdown_is_applied_side_by_side_only():
    m = view(with_openmm(tower()))
    a, b = md(engine="", name="slab"), S.PlanJob("crest", "crest", molecules=1000, size=40.0)
    x0 = S.side_by_side(a, b, m)
    x1 = S.side_by_side(a, b, m, slowdown=0.2)
    assert x1.serial_hours == pytest.approx(x0.serial_hours)
    assert x1.side_hours == pytest.approx(1.2 * x0.side_hours, rel=1e-6)
    assert x1.verdict == "one after the other" and "20% slowdown" in " ".join(x1.lines)


def test_two_gpu_jobs_never_share_one_gpu():
    m = view(with_openmm(tower()))
    x = S.side_by_side(md(engine="", name="one"), md(engine="", name="two"), m)
    assert sum(1 for a in x.side if a.gpu) <= 1


def test_an_unknown_time_is_said_plainly():
    m = view(with_openmm(tower()), cost_models={})
    x = S.side_by_side(md(engine=""), S.PlanJob("crest", "crest", molecules=10, size=40.0), m)
    assert x.verdict == "unknown"
    assert "can share" in x.lines[0] and "timings --fit" in x.lines[0]


# ----------------------------------------------------------------------------------------------- specs, folders, CLI
def test_job_specs():
    j = S.parse_job("md:atoms=9780,ns=13,dispersion=cutoff16,engine=openmm,box=40x40x200,name=C6F14")
    assert (j.kind, j.atoms, j.ns, j.dispersion, j.engine, j.box, j.name) == (
        "md", 9780, 13.0, "cutoff16", "openmm", (40.0, 40.0, 200.0), "C6F14")
    j = S.parse_job("build:method=AIMNet2,molecules=5000,solvent=water,atoms=30,threads=8,device=gpu")
    assert (j.method, j.solvent, j.molecules, j.size, j.threads, j.device) == ("aimnet2", "water", 5000, 30.0, 8, "gpu")
    j = S.parse_job("crest:molecules=1318,atoms=40,rotors=10,quick=no,method=gfn2,running=yes")
    assert j.crest_level() == "gfn2" and j.rotors == 10.0 and j.running
    for bad, msg in (("md:atoms=10,ns=1", "dispersion"), ("build:molecules=3", "method"),
                     ("crest:molecules=3", "atoms"), ("build:method=mmff,molecules=3,colour=red", "key of"),
                     ("build:method=mmff,molecules=-3", "0 or more"), ("md:atoms=1,ns=1,dispersion=foo", "dispersion"),
                     ("build:method=mmff,molecules=3,device=tpu", "device=cpu")):
        with pytest.raises(ValueError, match=msg):
            S.parse_job(bad)
    with pytest.raises(ValueError, match="not a folder"):
        S.parse_job("no/such/folder")


def _run_folder(tmp_path, state="paused", segments=()):
    from quantabench.md import sim

    stage = lambda name, a, b: sim.Stage(name, "production", a, b)                     # noqa: E731
    proto = sim.Protocol("liquid-slab", dispersion="cutoff16", parts=[
        sim.Part("bulk", "in.bulk", "log.bulk", stages=[stage("npt", 0, 1_000_000)]),
        sim.Part("slab", "in.slab", "log.slab", stages=[stage("nvt", 0, 12_000_000)])])
    run = sim.Run("liquid-slab", name="c6f14 test", protocol=proto, engine="lammps", state=state,
                  system=sim.System(n_atoms=9780))
    run.segments = [sim.Segment(**s) for s in segments]
    d = tmp_path / "slab"
    d.mkdir()
    run.save(d)
    return d


def test_md_job_from_a_run_folder(tmp_path):
    d = _run_folder(tmp_path, segments=[
        {"n": 1, "part": "bulk", "machine": "tower-test", "variant": "cpu-cutoff16", "threads": 8, "status": "done",
         "ns_per_day": 9.0, "last_step": 1_000_000},
        {"n": 2, "part": "slab", "machine": "tower-test", "variant": "gpu-cutoff16", "threads": 6, "status": "paused",
         "ns_per_day": 22.1, "end": {"kind": "data", "file": "slab.pause.data", "step": 3_000_000}}])
    j = S.parse_job(str(d))
    assert (j.kind, j.atoms, j.dispersion, j.engine, j.part) == ("md", 9780, "cutoff16", "lammps", "slab")
    assert j.ns == pytest.approx(9.0) and len(j.segments) == 2 and not j.running
    plan = S.plan_job(j, view(tower()))
    assert plan.best.source == "this run's segments"


def test_a_running_md_folder_keeps_its_segment(tmp_path):
    d = _run_folder(tmp_path, state="running", segments=[
        {"n": 1, "part": "bulk", "machine": "tower-test", "variant": "gpu-cutoff16", "threads": 6,
         "status": "running"}])
    j = S.md_job_from_folder(d)
    assert j.running and j.threads == 6 and j.device == "gpu" and any("until it pauses" in n for n in j.notes)


def test_an_unstarted_folder_may_switch_engine(tmp_path):
    j = S.md_job_from_folder(_run_folder(tmp_path, state="written"))
    assert j.engine == "" and j.ns == pytest.approx(13.0) and any("--engine" in n for n in j.notes)


def test_machine_view_reads_the_budget_and_finds_the_gpu():
    from quantabench import settings as sm

    s = sm.Settings()
    s.resources.free_threads = 3
    m = S.machine_view(tower(gpus=[]), settings=s)
    assert m.budget == 21 and m.physical == 12 and m.gpus == 1      # the GPU from its MD variants
    assert S.machine_view(tower(gpus=[], md_throughput={})).gpus == 0
    assert S.machine_view(tower(), threads=64).budget == 24


def test_the_cli_plans_from_the_registry(capsys):
    P.save_profile(with_openmm(tower()))
    from quantabench.dist import threads as T2

    path = T2.cost_model_path("tower-test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"machine": "tower-test", "models": {"gfnff quick": {
        "power": 3.9, "speedup_power": 0.5, "scale_s": 5e-4, "n_molecules": 700, "updated": "2026-11-27"}}}),
        encoding="utf-8")
    jobs = ["md:atoms=9780,ns=13,dispersion=cutoff16,name=slab", "crest:molecules=1000,atoms=40,name=crest"]
    assert cli.main(["jobs", "plan", *jobs, "--machine", "tower-test", "--threads", "23"]) == 0
    out = capsys.readouterr().out
    assert "slab (MD: 9,780 atoms" in out and "Fastest: GPU only" in out and "-> Run them side by side" in out
    assert cli.main(["jobs", "plan", *jobs, "--machine", "tower-test", "--threads", "23", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["budget"] == 23 and data["jobs"][0]["best"]["variant"] == "openmm-gpu-cutoff16"
    assert data["pairs"][0]["verdict"] == "side by side"
    assert data["jobs"][1]["best"]["source"] == "cost model"
    assert cli.main(["jobs", "plan", "build:molecules=3", "--machine", "tower-test"]) == 2
    assert cli.main(["jobs", "plan", jobs[0], "--machine", "nobody"]) == 2
    assert "not in the machine registry" in capsys.readouterr().err


def test_dialog_text():
    from types import SimpleNamespace

    cfg = SimpleNamespace(method="aimnet2", solvent="", conformer_search="rdkit", threads=0)
    text = S.dialog_text(cfg, 5000, tower(), budget=23)
    assert text.startswith("Devices here (23 threads): fastest GPU + 12 threads") and "short protocol" in text
    assert "no benchmark yet" in S.dialog_text(cfg, 10, None)
    crest = SimpleNamespace(method="gfn2-xtb", solvent="", conformer_search="crest", crest_method="gfnff",
                            crest_quick=True, threads=0)
    assert "CREST runs on the CPU only" in S.dialog_text(crest, 10, tower(), budget=23, size=40.0)


def test_two_jobs_with_the_same_title_are_told_apart():
    m = view(with_openmm(tower()))
    jobs = [S.parse_job("md:atoms=9780,ns=13,dispersion=cutoff16"), S.parse_job("md:atoms=9780,ns=13,dispersion=cutoff16")]
    plans, pairs = S.plan_queue(jobs, m)
    assert jobs[0].title() != jobs[1].title() and "(job 2)" in jobs[1].title()
    x = pairs[0]
    assert sorted(x.serial_each) == pytest.approx([plans[0].best.hours, 2 * plans[0].best.hours])
    assert sum(1 for a in x.side if a.gpu) <= 1
