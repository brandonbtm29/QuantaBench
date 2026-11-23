"""SIM-1: the run supervisor with a real LAMMPS (skipped without an ``lmp`` executable QuantaBench can find).

A tiny hexane slab (540 atoms, 1 thread, a few thousand steps per part): a whole run; a pause and a resume give one
continuous pressure-tensor series and density-profile series, with the same energies at the join; a stopped (killed)
segment resumes from its last periodic restart file; a run that moved to another build resumes from the pause data
file; a later segment can use other engine settings. ``QB_TEST_MD_GPU=1`` adds a CPU segment followed by a GPU one
where LAMMPS has the GPU package."""

import os
import re
import shutil

import numpy as np
import pytest

from quantabench.md import liquid as L
from quantabench.md import sim
from quantabench.md import supervisor as SV
from quantabench.md import surface_tension as S
from quantabench.md import telemetry as T

PRIORITY = "low"            # below normal, as the workstation's runs share it


@pytest.fixture(scope="module")
def lammps():
    from quantabench.md import bench

    info, why = bench.find_lammps(refresh=True)
    if info is None:
        pytest.skip(why)
    if not info.has("KSPACE") or not info.has("MOLECULE"):
        pytest.skip(f"LAMMPS {info.version} lacks KSPACE or MOLECULE")
    return info


def _config(**kw):
    base = dict(lateral=18.0, slab_thickness=18.0, packing="lattice", dispersion="cutoff8", thermo_every=100,
                warmup_ps=0.5, compress_ps=1.0, bulk_ns=0.003, settle_ns=0.001, production_ns=0.009,
                pressure_every=10, profile_every=10, profile_repeat=10, profile_freq=100, restart_every=1000,
                dump_every=500, vacuum=20.0)
    base.update(kw)
    return L.LiquidSlabConfig(**base)


def _supervisor(folder, info, **kw):
    kw.setdefault("threads", 1)
    return SV.Supervisor(folder, lammps=info.path, priority=PRIORITY, echo=None, progress_every_s=0.2, **kw)


def _progress(folder) -> dict:
    return sim.read_json(folder / sim.PROGRESS_FILE) or {}


@pytest.fixture(scope="module")
def slab_ready(lammps, tmp_path_factory):
    """A folder whose bulk part is done and whose slab segment was stopped at its start (no checkpoint)."""
    from quantabench.md import forcefield as F

    folder = tmp_path_factory.mktemp("md") / "hexane"
    tm = F.type_molecule("CCCCCC", "hexane", "oplsaa")
    L.write_liquid_slab(tm, folder, _config(), density_estimate=0.66)
    run = _supervisor(folder, lammps, should_stop=lambda: _progress(folder).get("part") == "slab").run()
    assert run.state == "stopped", [s.__dict__ for s in run.segments]
    assert [(s.part, s.status) for s in run.segments] == [("bulk", "done"), ("slab", "stopped")]
    bulk = run.segments[0]
    assert (bulk.first_step, bulk.last_step) == (0, 5000) and bulk.ns_per_day > 0
    assert bulk.build["version"] == lammps.version and bulk.threads == 1 and bulk.variant == "cpu-cutoff8"
    assert (folder / "bulk_final.data").is_file() and (folder / "log.bulk").is_file()
    return folder


def _copy(src, dst):
    shutil.copytree(src, dst)
    for name in (sim.LOCK_FILE, sim.STOP_FILE):
        if (dst / name).exists():
            (dst / name).unlink()
    return dst


def _series_ok(folder, end):
    """One pressure-tensor series every 10 steps from 0 to ``end`` and one density profile every 100."""
    a = np.loadtxt(folder / "pressure_tensor.txt", comments="#")
    assert a[0, 0] == 0 and a[-1, 0] == end and np.all(np.diff(a[:, 0]) == 10), np.unique(np.diff(a[:, 0]))
    steps, _z, rho = S.read_profiles(folder / "density_profile.txt")
    raw = [int(line.split()[0]) for line in open(folder / "density_profile.txt", encoding="utf-8")
           if line[:1].isdigit()]
    assert raw == list(range(100, end + 1, 100)) and len(steps) == len(raw)


def _row(log, step):
    for head, rows in T.read_thermo(log):
        for r in rows:
            if int(r[0]) == step and "PotEng" in head:
                return dict(zip(head, r))
    return None


def test_resume_runs_the_slab_to_the_end(slab_ready, tmp_path):
    folder = _copy(slab_ready, tmp_path / "run")
    info = _info()
    run = _supervisor(folder, info, resume=True).run()
    assert run.state == "done" and run.next_part() is None
    seg = run.segments[-1]
    assert (seg.part, seg.status, seg.start["kind"], seg.first_step, seg.last_step) == ("slab", "done", "start", 0,
                                                                                         10_000)
    _series_ok(folder, 10_000)
    prog = _progress(folder)
    assert prog["state"] == "done" and prog["ns_done"] == pytest.approx(run.ns_total)
    st = SV.read_status(folder)
    assert st["state"] == "done" and not st["alive"] and not (folder / sim.LOCK_FILE).exists()
    r = S.analyze(folder, skip_ns=0.0)
    assert np.isfinite(r["sigma"]["mean_mN_m"]) and [s["part"] for s in r["segments"]] == ["bulk", "slab", "slab"]
    assert SV.Supervisor(folder, echo=None, resume=True).run().state == "done"        # nothing left to do


def _info():
    from quantabench.md import bench

    return bench.find_lammps()[0]


def test_pause_and_resume_give_one_series(slab_ready, tmp_path):
    folder = _copy(slab_ready, tmp_path / "run")
    info = _info()

    def pause_once():
        p = _progress(folder)
        if p.get("part") == "slab" and (p.get("step") or 0) >= 1000 and not (folder / "STOP").exists():
            (folder / "STOP").write_text("pause", encoding="utf-8")
        return False

    run = _supervisor(folder, info, resume=True, should_stop=pause_once).run()
    seg = run.segments[-1]
    assert run.state == "paused" and seg.status == "paused", seg.__dict__
    s = seg.end["step"]
    assert s % 1000 == 0 and 1000 < s < 10_000 and seg.last_step == s
    assert sim.restart_step(folder / "slab.pause.restart") == s and sim.data_step(folder / "slab.pause.data") == s
    assert not (folder / "STOP").exists() and SV.read_status(folder)["state"] == "paused"
    # resume with other engine settings (2 threads): from the binary checkpoint, outputs appended
    run = _supervisor(folder, info, resume=True, threads=2).run()
    last = run.segments[-1]
    assert run.state == "done" and last.start == {"kind": "restart", "file": "slab.pause.restart", "step": s}
    assert last.threads == 2 and last.first_step == s and last.log == f"log.slab.seg{last.n}"
    assert "-pk omp 2" in " ".join(last.command)
    _series_ok(folder, 10_000)
    before, after = _row(folder / "log.slab", s), _row(folder / last.log, s)
    assert before and after and after["PotEng"] == pytest.approx(before["PotEng"], rel=1e-9, abs=1e-9)
    frames = [int(x) for x in re.findall(r"ITEM: TIMESTEP\n(\d+)", (folder / "slab.lammpstrj").read_text(encoding="utf-8"))]
    assert frames == list(range(0, 10_001, 500))


def test_a_stopped_segment_resumes_from_its_last_restart(slab_ready, tmp_path):
    folder = _copy(slab_ready, tmp_path / "run")
    info = _info()
    run = _supervisor(folder, info, resume=True,
                      should_stop=lambda: (_progress(folder).get("step") or 0) >= 2500
                      and _progress(folder).get("part") == "slab").run()
    stopped = run.segments[-1]
    assert run.state == "stopped" and stopped.status == "stopped" and stopped.end is None
    run = _supervisor(folder, info, resume=True).run()
    last = run.segments[-1]
    assert run.state == "done" and last.start["kind"] == "restart" and last.start["file"].startswith("slab.restart.")
    r = last.start["step"]
    assert r % 1000 == 0 and 1000 <= r <= stopped.last_step
    _series_ok(folder, 10_000)
    assert "outputs cut at step" in (folder / sim.SUPERVISOR_LOG).read_text(encoding="utf-8")


def test_a_run_moved_to_another_build_resumes_from_the_data_file(slab_ready, tmp_path):
    folder = _copy(slab_ready, tmp_path / "run")
    info = _info()
    (folder / "STOP").write_text("pause", encoding="utf-8")

    def keep_stop():          # the supervisor removes a STOP from before; put it back once the slab runs
        if _progress(folder).get("part") == "slab" and not (folder / "STOP").exists():
            (folder / "STOP").write_text("pause", encoding="utf-8")
        return False

    run = _supervisor(folder, info, resume=True, should_stop=keep_stop).run()
    assert run.state == "paused"
    s = run.segments[-1].end["step"]
    run = sim.Run.load(folder)
    run.segments[-1].build = {**run.segments[-1].build, "version": "another build"}
    run.save(folder)
    run = _supervisor(folder, info, resume=True).run()
    last = run.segments[-1]
    assert run.state == "done" and last.start == {"kind": "data", "file": "slab.pause.data", "step": s}
    text = (folder / last.input).read_text(encoding="utf-8")
    assert "read_data       slab.pause.data" in text and f"reset_timestep  {s}" in text
    _series_ok(folder, 10_000)


@pytest.mark.skipif(os.environ.get("QB_TEST_MD_GPU") != "1", reason="set QB_TEST_MD_GPU=1 on a machine with a GPU")
def test_a_cpu_segment_then_a_gpu_segment(slab_ready, tmp_path):
    from quantabench.md import bench

    info = _info()
    ok, why = bench.can_run(info, bench.Variant("gpu", "cutoff8"))
    if not ok:
        pytest.skip(why)
    folder = _copy(slab_ready, tmp_path / "run")

    def pause_once():
        p = _progress(folder)
        if p.get("part") == "slab" and (p.get("step") or 0) >= 1000 and not (folder / "STOP").exists():
            (folder / "STOP").write_text("pause", encoding="utf-8")
        return False

    run = _supervisor(folder, info, resume=True, variant="cpu", threads=2, should_stop=pause_once).run()
    s = run.segments[-1].end["step"]
    run = _supervisor(folder, info, resume=True, variant="gpu", threads=2).run()
    last = run.segments[-1]
    assert run.state == "done" and last.variant == "gpu-cutoff8" and last.start["step"] == s
    assert "-sf hybrid gpu omp" in " ".join(last.command)
    _series_ok(folder, 10_000)
    before, after = _row(folder / "log.slab", s), _row(folder / last.log, s)
    assert after["PotEng"] == pytest.approx(before["PotEng"], rel=1e-4)       # mixed precision on the GPU
