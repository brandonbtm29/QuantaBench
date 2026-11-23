"""SIM-1: MD runs as jobs, without LAMMPS. The run model (run.json), checkpoints and output cuts, the inputs' pause
lines and the segment inputs, the supervisor's choices, telemetry, and the md run/status/pause/resume commands."""

import json
import re
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from quantabench import cli
from quantabench.md import liquid as L
from quantabench.md import sim
from quantabench.md import supervisor as SV
from quantabench.md import surface_tension as S
from quantabench.md import telemetry as T


@pytest.fixture(scope="module")
def hexane():
    from quantabench.md import forcefield as F

    return F.type_molecule("CCCCCC", "hexane", "oplsaa")


def tiny_config(**kw) -> L.LiquidSlabConfig:
    base = dict(lateral=18.0, slab_thickness=18.0, packing="lattice", dispersion="cutoff8", thermo_every=100,
                warmup_ps=0.5, compress_ps=1.0, bulk_ns=0.003, settle_ns=0.001, production_ns=0.008,
                pressure_every=10, profile_every=10, profile_repeat=10, profile_freq=100, restart_every=1000,
                dump_every=500, vacuum=20.0)
    base.update(kw)
    return L.LiquidSlabConfig(**base)


def restart_bytes(step: int, units: bytes = b"real", flag48: bool = True) -> bytes:
    """A LAMMPS binary restart header as 22 Jul 2025 writes it (flags 0, 1, 48, 2, 3, 4 = UNITS, 5 = NTIMESTEP)."""
    b = b"LammpS RestartT\x00" + struct.pack("<ii", 1, 3)
    b += struct.pack("<ii", 0, 12) + b"22 Jul 2025\x00"
    b += struct.pack("<ii", 1, 4) + (struct.pack("<ii", 48, 4) if flag48 else b"")
    b += struct.pack("<ii", 2, 4) + struct.pack("<ii", 3, 8)
    b += struct.pack("<ii", 4, len(units) + 1) + units + b"\x00"
    b += struct.pack("<iq", 5, step) + struct.pack("<ii", 6, 3) + b"\x00" * 64
    return b


# ------------------------------------------------------------------------------------------- the model
def test_protocol_of_a_liquid_slab_matches_its_inputs():
    cfg = L.LiquidSlabConfig(production_ns=10.0).resolved()
    proto = L.protocol_for(cfg, compress=True)
    bulk, slab = proto.part("bulk"), proto.part("slab")
    assert [s.name for s in bulk.stages] == ["minimize", "warmup", "compress", "npt"]
    assert (bulk.stages[1].end, bulk.stages[2].end, bulk.stages[3].end) == (20_000, 70_000, 1_070_000)
    assert bulk.stages[1].timestep_fs == 0.5 and bulk.ns == pytest.approx(0.01 + 0.05 + 1.0)
    assert slab.last_step == 12_000_000 and slab.ns == pytest.approx(12.0) and proto.ns == pytest.approx(13.06)
    assert slab.needs == ["bulk_final.data", "ff.params"] and slab.done_tag == "QB_SLAB done"
    kinds = {o.name: (o.kind, o.rewrites_start) for o in slab.outputs}
    assert kinds == {"pressure_tensor.txt": ("ave/time", True), "density_profile.txt": ("ave/chunk", False),
                     "slab.lammpstrj": ("dump", True)}
    assert bulk.outputs[0].rewrites_start is False                  # ave/time 100 10 1000: no row at a resume's start
    assert proto.pause_every == 10_000 and proto.restart_every == 100_000
    assert slab.ns_at(3_000_000) == pytest.approx(3.0) and slab.steps_left(3_000_000) == 9_000_000
    assert bulk.stage_at(30_000).name == "compress" and bulk.ns_at(None) == 0.0


def test_pause_every_follows_the_output_blocks():
    assert L.LiquidSlabConfig().resolved().pause_every == 10_000
    assert L.LiquidSlabConfig(profile_freq=20_000, restart_every=200_000).resolved().pause_every == 20_000
    assert L.LiquidSlabConfig(pressure_every=7, profile_freq=700, restart_every=7000).resolved().pause_every == 7000
    with pytest.raises(ValueError, match="multiple of 10000"):
        L.LiquidSlabConfig(pause_every=5000).resolved()
    with pytest.raises(ValueError, match="restart_every"):
        L.LiquidSlabConfig(restart_every=15_000).resolved()


def test_run_record_round_trips_and_tracks_progress(hexane, tmp_path):
    s = L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    assert (tmp_path / "run.json") in s.files
    run = sim.Run.load(tmp_path)
    assert run.template == "liquid-slab" and run.state == "written" and not run.legacy
    assert run.system.components[0]["count"] == s.n_molecules and run.system.n_atoms == s.n_atoms
    assert run.parameterization.physics["vdw"]["cutoff_A"] == 8.0
    assert run.variant == "" and run.protocol.dispersion == "cutoff8"     # a cutoff: auto may pick the GPU
    assert sim.Run.from_dict(json.loads(json.dumps(run.to_dict()))).to_dict() == run.to_dict()
    assert run.next_part().name == "bulk" and run.ns_total == pytest.approx(0.0045 + 0.009)
    run.segments.append(sim.Segment(1, "bulk", status="done", first_step=0, last_step=5000))
    run.segments.append(sim.Segment(2, "slab", status="paused", first_step=0, last_step=3000,
                                    end={"kind": "restart", "file": "slab.pause.restart", "step": 3000}))
    assert run.next_part().name == "slab" and run.part_done("bulk")
    assert run.ns_done() == pytest.approx(0.0045 + 0.003)
    assert run.steps_left() == [(6000, 1.0)]
    assert run.ns_done("slab", 4000) == pytest.approx(0.0085)
    # a newer QuantaBench's keys are ignored
    d = run.to_dict()
    d["future"] = 1
    d["segments"][0]["future"] = 2
    assert sim.Run.from_dict(d).segments[0].status == "done"


def test_restart_and_data_steps(tmp_path):
    for step, flag48 in ((0, True), (3_000_000, True), (12, False)):
        (tmp_path / "r").write_bytes(restart_bytes(step, flag48=flag48))
        assert sim.restart_step(tmp_path / "r") == step
    (tmp_path / "lj").write_bytes(restart_bytes(250, units=b"lj"))
    assert sim.restart_step(tmp_path / "lj") == 250
    (tmp_path / "bad").write_bytes(b"LammpS RestartT\x00" + b"\x00" * 10)
    assert sim.restart_step(tmp_path / "bad") is None and sim.restart_step(tmp_path / "none") is None
    (tmp_path / "x.data").write_text("LAMMPS data file via write_data, version 22 Jul 2025, timestep = 3000, units = "
                                     "real\n\n9 atoms\n", encoding="utf-8")
    assert sim.data_step(tmp_path / "x.data") == 3000
    (tmp_path / "y.data").write_text("hexane liquid (QuantaBench)\n", encoding="utf-8")
    assert sim.data_step(tmp_path / "y.data") is None


def _ave_time(path: Path, steps) -> None:
    rows = "".join(f"{s} {s * 0.1:g} 2 3\n" for s in steps)
    path.write_text(f"# Time-averaged data for fix ptens\n# TimeStep c1 c2 c3\n{rows}", encoding="utf-8")


def test_cut_at_keeps_one_series(tmp_path):
    p = tmp_path / "pressure_tensor.txt"
    _ave_time(p, range(0, 1010, 10))
    assert sim.cut_at(p, "ave/time", 500, keep_step=False) == 51
    steps = np.loadtxt(p, comments="#")[:, 0]
    assert steps[-1] == 490 and len(steps) == 50
    with open(p, "a", encoding="utf-8") as fh:              # a resumed fix appends its header again, then step 500
        fh.write("# Time-averaged data for fix ptens\n# TimeStep c1 c2 c3\n")
        fh.writelines(f"{s} 9 9 9\n" for s in range(500, 600, 10))
    a = S.read_pressure_tensor(p)
    assert np.all(np.diff(a[:, 0]) == 10) and a[-1, 0] == 590
    # an ave/chunk file: blocks; the one at the step stays (a block whose window started before is not rewritten)
    c = tmp_path / "density_profile.txt"
    head = "# Chunk-averaged data for fix dprof\n# Timestep Number-of-chunks Total-count\n# Chunk Coord1 Ncount m\n"
    blocks = "".join(f"{s} 2 10\n  1 0.25 5 0.6\n  2 0.75 5 0.7\n" for s in range(100, 1100, 100))
    c.write_text(head + blocks, encoding="utf-8")
    assert sim.cut_at(c, "ave/chunk", 500, keep_step=True) == 5
    st, z, rho = S.read_profiles(c)
    assert list(st) == [100, 200, 300, 400, 500] and rho.shape == (5, 2)
    # a dump: frames
    d = tmp_path / "slab.lammpstrj"
    d.write_text("".join(f"ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n1\nITEM: ATOMS id x\n1 0.0\n"
                         for s in range(0, 2000, 500)), encoding="utf-8")
    assert sim.cut_at(d, "dump", 1000, keep_step=False) == 2
    assert re.findall(r"ITEM: TIMESTEP\n(\d+)", d.read_text(encoding="utf-8")) == ["0", "500"]
    assert sim.cut_at(tmp_path / "missing.txt", "ave/time", 5, keep_step=False) == 0
    assert sim.cut_at(d, "print", 0, keep_step=False) == 0


def test_readers_merge_by_step(tmp_path):
    p = tmp_path / "pressure_tensor.txt"
    _ave_time(p, [0, 10, 20, 30])
    with open(p, "a", encoding="utf-8") as fh:             # a hand-made resume from step 20 that was not cut
        fh.write("# TimeStep c1 c2 c3\n20 5 5 5\n30 6 6 6\n40 7 7 7\n")
    a = S.read_pressure_tensor(p)
    assert list(a[:, 0]) == [0, 10, 20, 30, 40] and list(a[2:, 1]) == [5, 6, 7]
    assert sim.merge_by_step([(2, "b"), (1, "a"), (2, "c")]) == [(1, "a"), (2, "c")]


# ------------------------------------------------------------------------------------------- the inputs
SEGMENT_COMMANDS = {"read_restart", "label", "write_restart", "quit"}


def _check_syntax(text: str) -> None:
    from test_md_liquid import LAMMPS_COMMANDS

    allowed = LAMMPS_COMMANDS | SEGMENT_COMMANDS
    cont = False
    for line in text.splitlines():
        s = line.split("#")[0].strip() if '"' not in line else line.strip()
        if not s:
            cont = False
            continue
        if not cont:
            assert s.split()[0] in allowed, line
        assert line.count('"') % 2 == 0, line
        cont = s.endswith("&")
    for v in set(re.findall(r"\$\{(\w+)\}", text)):
        assert re.search(rf"variable\s+{v}\s", text), f"${{{v}}} is used but never defined"
    for label in re.findall(r"jump SELF (\w+)", text):
        assert re.search(rf"^label\s+{label}$", text, re.M), label


def test_inputs_pause_through_a_stop_file(hexane, tmp_path):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    bulk = (tmp_path / "in.bulk").read_text(encoding="utf-8")
    slab = (tmp_path / "in.slab").read_text(encoding="utf-8")
    for text, part in ((bulk, "bulk"), (slab, "slab")):
        assert "variable        qb_stop equal is_file(STOP)" in text
        assert "fix             qb_halt all halt ${pause_every} v_qb_stop != 0 error soft" in text
        assert f"write_restart {part}.pause.restart" in text and f"QB_PAUSED {part} step" in text
        assert "variable pause_every    index 1000" in text
        _check_syntax(text)
    assert bulk.count("QB_PAUSED") == 3 and slab.count("QB_PAUSED") == 1          # after every run
    assert "run             ${npt_end} upto" in bulk and "run             ${slab_end} upto" in slab
    assert "restart         1000 bulk.restart.a bulk.restart.b" in bulk
    assert bulk.index("qb_halt") > bulk.index("qb_epair")       # the halt after the guard's run 0
    readme = (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert "quantabench-cli md pause" in readme and "RUN AS A JOB" in readme


def test_segment_inputs_continue_from_a_checkpoint(hexane, tmp_path):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    name = L.write_segment_input(tmp_path, "bulk", sim.Checkpoint("restart", "bulk.restart.b", 3000), 2)
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert name == "in.bulk.seg2" and "log             log.bulk.seg2" in text
    assert text.index("read_restart    bulk.restart.b") < text.index("pair_style      lj/cut/coul/long ${rc}") \
        < text.index("include         ff.params")                         # the new segment's suffix applies
    assert "tail yes" in text and "read_data" not in text and "velocity" not in text
    assert 'if "$(step) >= ${warmup_end}" then "jump SELF qb_after_warmup"' in text
    assert "label           qb_after_npt" in text and "append bulk_density.txt" in text
    _check_syntax(text)
    name = L.write_segment_input(tmp_path, "slab", sim.Checkpoint("data", "slab.pause.data", 4000), 5)
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert "read_data       slab.pause.data" in text and "reset_timestep  4000" in text
    assert "units           real" in text and "tail yes" not in text and "change_box" not in text
    assert "append pressure_tensor.txt" in text and "append density_profile.txt" in text
    assert '"dump_modify traj append yes"' in text and "QB_SLAB done" in text
    _check_syntax(text)
    with pytest.raises(ValueError, match="needs its step"):
        L.resume_input(hexane, tiny_config().resolved(), "slab", "x.data", kind="data")


def test_smd_folder_has_a_run_record(tmp_path):
    from ase.build import molecule

    from quantabench.build import surfaces
    from quantabench.md import lammps as M

    slab = surfaces.graphite_slab(4, 3, layers=1, vacuum=15)
    system = surfaces.place_on_surface(slab, molecule("CH4"), height=3.5)
    M.write_smd_setup(system, tmp_path, soft_steps=100, equil_steps=200, smd_steps=300,
                      charges=np.zeros(len(system)))
    run = sim.Run.load(tmp_path)
    part = run.protocol.part("smd")
    assert run.template == "smd" and part.resume == "rerun" and part.log == "log.lammps"
    assert [s.end for s in part.stages] == [100, 300, 600] and run.variant == "cpu"
    text = (tmp_path / "in.smd").read_text(encoding="utf-8")
    assert "fix             qb_halt all halt ${pause_every} v_qb_stop != 0 error soft" in text
    assert text.count("QB_PAUSED smd") == 3 and 'print           "QB_SMD done"' in text
    assert SV.choose_checkpoint(tmp_path, run, part, "m", {}).kind == "start"


# ------------------------------------------------------------------------------------------- the supervisor's choices
def _segment(n, part, status, machine="here", version="v1", **kw):
    return sim.Segment(n, part, machine=machine, build={"version": version, "path": "/lmp"}, status=status, **kw)


def test_choose_checkpoint(hexane, tmp_path):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    run = sim.Run.load(tmp_path)
    slab = run.protocol.part("slab")
    build = {"version": "v1", "path": "/lmp"}
    assert SV.choose_checkpoint(tmp_path, run, slab, "here", build).kind == "start"     # no segment yet
    run.segments = [_segment(1, "bulk", "done"), _segment(2, "slab", "stopped", first_step=0, last_step=2500)]
    assert SV.choose_checkpoint(tmp_path, run, slab, "here", build).kind == "start"     # no checkpoint
    (tmp_path / "slab.restart.a").write_bytes(restart_bytes(2000))
    (tmp_path / "slab.restart.b").write_bytes(restart_bytes(1000))
    c = SV.choose_checkpoint(tmp_path, run, slab, "here", build)
    assert (c.kind, c.file, c.step) == ("restart", "slab.restart.a", 2000)
    # a pause checkpoint further on wins; the binary one when this machine and build wrote it
    (tmp_path / "slab.pause.restart").write_bytes(restart_bytes(3000))
    (tmp_path / "slab.pause.data").write_text("LAMMPS data file via write_data, version x, timestep = 3000, "
                                              "units = real\n", encoding="utf-8")
    run.segments[-1] = _segment(2, "slab", "paused", first_step=0, last_step=3000)
    c = SV.choose_checkpoint(tmp_path, run, slab, "here", build)
    assert (c.kind, c.step) == ("restart", 3000)
    # another machine, or another LAMMPS build: only the portable data file
    c = SV.choose_checkpoint(tmp_path, run, slab, "elsewhere", build)
    assert (c.kind, c.file, c.step) == ("data", "slab.pause.data", 3000)
    assert SV.choose_checkpoint(tmp_path, run, slab, "here", {"version": "v2", "path": "/lmp"}).kind == "data"
    # a restart beyond the part's end is not this part's
    (tmp_path / "slab.restart.b").write_bytes(restart_bytes(10**9))
    assert SV.choose_checkpoint(tmp_path, run, slab, "here", build).step == 3000


def _lammps(gpu=True, api="OpenCL"):
    from quantabench.md import bench

    pk = ["KSPACE", "MOLECULE", "OPENMP"] + (["GPU"] if gpu else [])
    return bench.LammpsInfo("/opt/lmp", "test", "4 Jul 2026", pk, {"GPU": {"api": api}} if gpu else {}, gpu or None)


def test_choose_accelerator_and_command(hexane, tmp_path, monkeypatch):
    from quantabench.md import long_range as LR

    monkeypatch.setattr(LR, "_gpu_check_failed", lambda: False)
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    run = sim.Run.load(tmp_path)
    assert SV.choose_accelerator(run, _lammps(), None)[0] == "gpu"
    assert SV.choose_accelerator(run, _lammps(gpu=False), "auto")[0] == "cpu"
    assert SV.choose_accelerator(run, _lammps(), "cpu-cutoff8")[0] == "cpu"
    with pytest.raises(sim.RunError, match="dispersion is cutoff8"):
        SV.choose_accelerator(run, _lammps(), "gpu-cutoff16")
    with pytest.raises(sim.RunError, match="no GPU package"):
        SV.choose_accelerator(run, _lammps(gpu=False), "gpu")
    monkeypatch.setattr(LR, "_gpu_check_failed", lambda: True)
    assert "pressure check" in SV.choose_accelerator(run, _lammps(), None)[1]
    run.protocol.dispersion = "ewald"
    assert SV.choose_accelerator(run, _lammps(), None)[0] == "cpu"
    cmd, n, _label = SV.engine_command(_lammps(), "in.slab.seg3", "gpu", 6, None)
    assert cmd == ["/opt/lmp", "-nocite", "-log", "none", "-sf", "hybrid", "gpu", "omp", "-pk", "gpu", "1",
                   "device_type", "generic", "pair/only", "on", "-pk", "omp", "6", "-in", "in.slab.seg3"] and n == 6
    cmd, n, _label = SV.engine_command(_lammps(), "in.smd", "cpu", 4, "log.lammps")
    assert cmd[2:4] == ["-log", "log.lammps"] and cmd[4:8] == ["-sf", "omp", "-pk", "omp"]


def test_status_lock_and_pause(hexane, tmp_path, capsys):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    st = SV.read_status(tmp_path)
    assert st["state"] == "written" and not st["alive"] and st["ns_done"] == 0.0
    with pytest.raises(sim.RunError, match="not running"):
        SV.request_pause(tmp_path)
    assert not (tmp_path / sim.STOP_FILE).exists()
    run = sim.Run.load(tmp_path)
    run.state = "running"                      # a supervisor that died: no lock
    run.save(tmp_path)
    st = SV.read_status(tmp_path)
    assert st["state"] == "stopped" and st["stale"]
    with SV.held_lock(tmp_path):
        assert SV.lock_holder(tmp_path)["pid"] > 0
        with pytest.raises(sim.RunError, match="already runs it"):
            with SV.held_lock(tmp_path):
                pass
        assert "Pausing" in SV.request_pause(tmp_path)
        assert SV.read_stop(tmp_path) == "pause"
        SV.request_pause(tmp_path, now=True)
        assert SV.read_stop(tmp_path) == "now"
        assert SV.read_status(tmp_path)["stop_requested"] == "now"
    assert not (tmp_path / sim.LOCK_FILE).exists() and SV.clear_stop(tmp_path)
    (tmp_path / sim.LOCK_FILE).write_text(json.dumps({"pid": 2**30, "machine": SV.machine_name()}), encoding="utf-8")
    with SV.held_lock(tmp_path):              # a stale lock (its process is gone) is taken over
        pass
    # the commands
    assert cli.main(["md", "status", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "stopped (no supervisor is running it" in out and "part bulk: to do" in out
    assert cli.main(["md", "pause", str(tmp_path)]) == 1
    assert cli.main(["md", "run", str(tmp_path / "nothing")]) == 2
    assert cli.main(["md", "status", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"


def test_a_folder_written_before_sim1_gets_a_run_record(hexane, tmp_path):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    (tmp_path / "run.json").unlink()
    meta = json.loads((tmp_path / "liquid_setup.json").read_text(encoding="utf-8"))
    del meta["config"]["pause_every"]
    meta["config"]["restart_every"] = 1_000_000
    (tmp_path / "liquid_setup.json").write_text(json.dumps(meta), encoding="utf-8")
    assert sim.is_run_folder(tmp_path)
    run = SV.load_run(tmp_path)
    assert run.legacy and "STOP" in run.notes[0] and (tmp_path / "run.json").is_file()
    assert run.protocol.restart_every == 1_000_000 and run.protocol.pause_every == 1000
    assert not sim.is_run_folder(tmp_path / "x")


def test_supervisor_refuses_what_it_cannot_run(hexane, tmp_path, monkeypatch):
    L.write_liquid_slab(hexane, tmp_path, tiny_config(), density_estimate=0.66)
    monkeypatch.setattr(SV, "find_engine", lambda path=None: (None, "LAMMPS not found (test)"))
    with pytest.raises(sim.RunError, match="LAMMPS not found"):
        SV.Supervisor(tmp_path, echo=None).run()
    assert not (tmp_path / sim.LOCK_FILE).exists() and sim.Run.load(tmp_path).state == "written"
    run = sim.Run.load(tmp_path)
    run.segments.append(_segment(1, "bulk", "paused"))
    run.state = "paused"
    run.save(tmp_path)
    with pytest.raises(sim.RunError, match="md resume"):
        SV.Supervisor(tmp_path, echo=None).run()
    run.state, run.segments = "done", [_segment(1, "bulk", "done"), _segment(2, "slab", "done")]
    run.save(tmp_path)
    assert SV.Supervisor(tmp_path, echo=None, resume=True).run().state == "done"
    assert "already done" in (tmp_path / sim.SUPERVISOR_LOG).read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------- telemetry
LOG_HEAD = """LAMMPS (22 Jul 2025 - Update 4)
units real
   Step          Time           Temp          PotEng
         0   0              298.15         -100
"""


def test_log_tail_reads_only_what_was_appended(tmp_path):
    p = tmp_path / "log.slab"
    p.write_text(LOG_HEAD, encoding="utf-8")
    tail = T.LogTail(p)
    assert tail.poll() == 1 and tail.state.last_step == 0 and tail.state.header[2] == "Temp"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("       100   100            300.5          -101\n       200   200")      # a partial line
    assert tail.poll() == 1 and tail.state.last_step == 100
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("            299.0          -102\n"
                 "Loop time of 2.5 on 1 procs for 200 steps with 540 atoms\n\n"
                 "Performance: 6.912 ns/day, 3.472 hours/ns, 80.000 timesteps/s, 43.200 katom-step/s\n"
                 "WARNING: something harmless\nQB_PAUSED slab step 200\nERROR: Lost atoms: original 540 current 539\n")
    assert tail.poll() == 1
    st = tail.state
    assert (st.first_step, st.last_step, st.paused_step, st.warnings) == (0, 200, 200, 1)
    assert st.best_run["ns_per_day"] == 6.912 and st.best_run["steps"] == 200
    assert st.errors == ["ERROR: Lost atoms: original 540 current 539"] and st.value("Temp") == 299.0
    p.write_text(LOG_HEAD, encoding="utf-8")                  # rewritten (a rerun): starts over
    tail.poll()
    assert tail.state.last_step == 0 and not tail.state.errors
    assert T.summarize_log(p).first_step == 0 and len(T.read_thermo(p)) == 1


def test_speed_eta_and_alerts():
    m = T.SpeedMeter(window_s=100)
    for i in range(5):
        m.add(i * 1000, t=1000.0 + i * 10)
    assert m.steps_per_s() == pytest.approx(100.0) and m.ns_per_day(1.0) == pytest.approx(8.64)
    m.add(0, t=1100.0)                                         # a new segment: the counter went back
    assert m.steps_per_s() is None
    assert T.eta_seconds([(9000, 1.0), (1000, 0.5)], 100.0) == pytest.approx(100.0)
    assert T.eta_seconds([(1, 1.0)], None) is None
    assert T.format_eta(30) == "30 s" and T.format_eta(3 * 3600 + 120) == "3 h 02 min" and T.format_eta(None) == ""
    st = T.LogState(header=["Step", "Temp"])
    for i in range(30):
        st.rows.append([i, 340.0])
    st.grew = time.time() - 4000
    st.errors.append("QB_ERROR: no energy")
    a = T.alerts(st, target_T=298.15, running=True, stall_s=1800)
    assert any("temperature off target: 340.0 K" in x for x in a) and any("has not grown" in x for x in a)
    assert a[0].startswith("error: QB_ERROR")
    assert not T.alerts(T.LogState(), target_T=298.15)


def test_md_run_modules_are_pure_python():
    import os
    import subprocess
    import sys

    code = ("import sys; import quantabench.md.sim, quantabench.md.telemetry, quantabench.md.supervisor; "
            "print(sorted(m for m in ('numpy', 'rdkit', 'lammps', 'PySide6') if m in sys.modules))")
    src = str(Path(sim.__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
