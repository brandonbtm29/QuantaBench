"""DIST-7: machine control. Pausing a machine (drain / now), its work moving to the other machines, a machine that
stops answering (a closing laptop), its return, the first-done merge, the split choices and the settings merge."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import pytest

from quantabench import settings as settings_mod
from quantabench.chem import jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import client as client_mod
from quantabench.dist import control
from quantabench.dist import worker as worker_mod
from quantabench.dist.client import MachineWatch, Share, WorkerClient

TOKEN = "test-token-" + "z" * 32
MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F"),
        ("methanol", "CO"), ("acetone", "CC(C)=O"), ("propanol", "CCCO"), ("butane", "CCCC"),
        ("chloroform", "ClC(Cl)Cl"), ("dmso", "CS(C)=O"), ("acetonitrile", "CC#N"), ("formic_acid", "OC=O")]
CONFIG = {"method": "mmff", "max_conformers": 2, "refine_top": 1, "xtb_properties": False, "formats": ["xyz"],
          "rank_method": "refine"}
VERSION = worker_mod.__version__


def inputs(n=None):
    return [from_smiles(s, id=i) for i, s in MOLS[:n]]


def wait_for(cond, timeout=60.0, what="condition"):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def summary_ids(folder):
    with open(Path(folder) / "summary.csv", newline="", encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f)]


def stoppable(monkeypatch, seconds, ids=None):
    """pipeline.run_molecule takes ``seconds`` (only ``ids`` when given) and honours should_stop meanwhile: a stopped
    molecule comes back 'cancelled', as a real one does at its next checkpoint. Returns the ids started."""
    real = pipeline.run_molecule
    started: list[str] = []

    def run(inp, cfg, device="cpu", threads=1, should_stop=lambda: False, log=lambda m: None, stage=lambda s: None):
        started.append(inp.id)
        if ids is None or inp.id in ids:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if should_stop():
                    return pipeline.MoleculeResult(input=inp, status="cancelled", error="cancelled")
                time.sleep(0.01)
        return real(inp, cfg, device, threads, should_stop, log, stage=stage)

    monkeypatch.setattr(pipeline, "run_molecule", run)
    return started


@pytest.fixture
def qb_settings(isolated_settings, monkeypatch):
    s = settings_mod.Settings(output_dir=str(isolated_settings / "client_runs"))
    s.resources.threads, s.resources.gpu = 1, "off"
    s.dist.worker_token = TOKEN
    settings_mod.save(s)
    monkeypatch.setattr("quantabench.chem.properties.available", lambda: False)  # no dxtb step in tests
    return s


@pytest.fixture
def make_worker(tmp_path, qb_settings):
    started = []

    def make(name, output=None, **kw):
        w = worker_mod.Worker("127.0.0.1", 0, TOKEN, output or tmp_path / f"{name}_runs",
                              state_dir=tmp_path / f"{name}_state", name=name, awake=False, **kw)
        started.append(w)
        return w.start()

    yield make
    for w in started:
        w.stop(timeout=30)


def cfg_of(**kw):
    return pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                                write_files=False, rank_method="refine", **kw)


# ------------------------------------------------------------------------------------- BatchControl
def test_sequential_batch_holds_drains_and_stops_now(tmp_path, monkeypatch):
    started = stoppable(monkeypatch, 0.3)
    ctl = jobs.BatchControl()
    run_dir = tmp_path / "run"
    out: dict = {}
    th = threading.Thread(target=lambda: out.setdefault("r", jobs.run_resumable(inputs(6), cfg_of(), run_dir,
                                                                                control=ctl)))
    ctl.set(mode="hold")
    th.start()
    time.sleep(0.6)
    assert started == []                                   # held from the start: nothing begins
    ctl.set(mode="run")
    wait_for(lambda: len(started) >= 2, what="two molecules")
    ctl.set(mode="drain", interrupt=True)                 # pause now: the running one stops at a checkpoint
    th.join(60)
    st = jobs.open_run(run_dir)
    assert st.finished >= 1 and st.remaining >= 1 and st.counts["running"] == 0
    with jobs.JobStore(run_dir) as store:
        stopped = [i for i, s in store.statuses().items() if s == "cancelled"]
    assert len(stopped) == 1 and stopped[0] == started[-1] and not ctl.interrupt
    # the rest runs on the next call, the stopped molecule included
    jobs.run_resumable(None, cfg_of(), run_dir)
    st = jobs.open_run(run_dir)
    assert st.finished == 6 and st.remaining == 0


def test_hold_now_puts_the_molecule_back_and_resume_runs_it(tmp_path, monkeypatch):
    started = stoppable(monkeypatch, 0.4)
    ctl = jobs.BatchControl()
    run_dir = tmp_path / "run"
    th = threading.Thread(target=lambda: jobs.run_resumable(inputs(3), cfg_of(), run_dir, control=ctl))
    th.start()
    wait_for(lambda: started, what="a start")
    ctl.set(mode="hold", interrupt=True)
    wait_for(lambda: not ctl.interrupt, what="the interrupt")
    n = len(started)
    time.sleep(0.5)
    assert len(started) == n                                 # held: nothing new
    ctl.set(mode="run")
    th.join(60)
    st = jobs.open_run(run_dir)
    assert st.finished == 3 and started.count(started[0]) == 2   # the stopped molecule ran again


def test_control_source_follows_this_machines_pause_file(tmp_path):
    control.write_pause(tmp_path, True, "now", by="dock")
    ctl = jobs.BatchControl(source=lambda: control.pause_gate(tmp_path), interval=0)
    assert ctl.poll() == "hold" and ctl.interrupt
    control.write_pause(tmp_path, False)
    assert ctl.poll() == "run" and not ctl.interrupt
    assert control.read_pause(tmp_path / "missing") == {"paused": False, "mode": "drain", "since": "", "by": "",
                                                        "reason": ""}
    (tmp_path / control.PAUSE_FILE).write_text("{torn", encoding="utf-8")
    assert control.read_pause(tmp_path)["paused"] is False


def test_gui_batch_follows_this_machines_pause(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench.gui.workers import BuildWorker

    started = stoppable(monkeypatch, 0.1)
    control.write_pause(None, True, "drain", by="dock")
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                               rank_method="refine", output_dir=str(tmp_path))
    bw = BuildWorker(inputs(3), cfg, lanes="1", run_dir=tmp_path / "gui_batch")
    th = threading.Thread(target=bw.run)          # its signals are queued for the GUI thread: read the run folder
    th.start()
    time.sleep(1.0)
    assert started == [] and th.is_alive() and bw.control.mode == "hold"
    control.write_pause(None, False)
    th.join(60)
    assert not th.is_alive() and len(started) == 3
    assert jobs.open_run(tmp_path / "gui_batch").finished == 3


def test_lanes_hold_and_drain(tmp_path, monkeypatch):
    from quantabench.dist import lanes as L

    monkeypatch.setenv(L.TEST_DELAY_ENV, "cpu1=0.6,cpu2=0.6")
    ctl = jobs.BatchControl()
    run_dir = tmp_path / "lanes"
    lanes = [L.Lane("cpu1", "cpu", 1), L.Lane("cpu2", "cpu", 1)]
    logs: list[str] = []
    out: dict = {}
    th = threading.Thread(target=lambda: out.setdefault("r", jobs.run_resumable(inputs(10), cfg_of(), run_dir,
                                                                                lanes=lanes, control=ctl,
                                                                                log=logs.append)))
    th.start()
    def finished():
        try:
            return jobs.open_run(run_dir).finished
        except Exception:          # the table is being created
            return 0

    wait_for(lambda: finished() >= 2, timeout=120, what="two molecules")
    ctl.set(mode="hold")
    wait_for(lambda: any(line.startswith("Paused (hold)") for line in logs), what="the hold")
    time.sleep(1.5)                                 # running ones finish; nothing new starts
    a = jobs.open_run(run_dir).finished
    time.sleep(1.0)
    assert jobs.open_run(run_dir).finished == a and th.is_alive()
    ctl.set(mode="drain")
    th.join(120)
    st = jobs.open_run(run_dir)
    assert st.remaining > 0 and st.counts["running"] == 0 and any("stay in the job table" in x for x in logs)


# ------------------------------------------------------------------------------------- the worker
def test_worker_pause_drain_release_and_resume(make_worker, monkeypatch):
    started = stoppable(monkeypatch, 0.25)
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    job = c.submit(inputs(8), dict(CONFIG), "pz")
    wait_for(lambda: len(started) >= 2, what="two molecules")
    q = c.pause(True, "drain", by="test")
    assert q["paused"] and q["pause"]["mode"] == "drain" and q["pause"]["by"] == "test"
    wait_for(lambda: c.status(job)["state"] == "paused", what="the paused state")
    st = c.status(job)
    assert st["machine_paused"] and st["counts"]["pending"] >= 1 and st["counts"]["running"] == 0
    assert control.read_pause(w.manager.pause_dir)["paused"]            # saved: survives a restart
    # a paused job hands its unstarted molecules over from its job table
    mols = c.molecules(job)
    pending = [i for i, s in mols.items() if s == "pending"]
    ids = c.release(job, 2, "w2")
    assert len(ids) == 2 and set(ids) <= set(pending)
    assert c.molecules(job)[ids[0]] == "moved"
    q = c.pause(False)
    assert not q["paused"]
    st = c.wait(job, poll=0.1, timeout=120)
    assert st["state"] == "done" and st["counts"]["moved"] == 2 and st["counts"]["ok"] == 6


def test_worker_pause_now_stops_running_molecules(make_worker, monkeypatch):
    started = stoppable(monkeypatch, 5.0, ids={"ethanol"})
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    job = c.submit(inputs(3), dict(CONFIG), "now")
    wait_for(lambda: started, what="the first molecule")
    t0 = time.monotonic()
    c.pause(True, "now")
    wait_for(lambda: c.status(job)["state"] == "paused", timeout=20, what="paused")
    assert time.monotonic() - t0 < 4.0                                     # did not wait 5 s for ethanol
    states = c.molecules(job)
    assert states["ethanol"] == "cancelled"                             # not started, as far as anyone is concerned
    # everything not started moves away: the paused job is then done, with its run folder finished
    rest = c.release(job, 100, "w2")
    assert sorted(rest) == sorted(i for i, s in states.items() if s in ("pending", "cancelled"))
    st = c.status(job)
    assert st["state"] == "done" and st["counts"]["moved"] == len(rest)
    assert (Path(st["run_dir"]) / worker_mod.JOB_FILE).is_file()


def test_paused_worker_stays_paused_across_a_restart(tmp_path, qb_settings, monkeypatch):
    stoppable(monkeypatch, 0.2)
    state, out = tmp_path / "state", tmp_path / "runs"
    w = worker_mod.Worker("127.0.0.1", 0, TOKEN, out, state_dir=state, name="w1", awake=False).start()
    c = WorkerClient(w.url, TOKEN)
    c.pause(True)
    job = c.submit(inputs(2), dict(CONFIG), "rs")
    time.sleep(0.5)
    assert c.status(job)["state"] == "queued"
    w.stop(10)
    w = worker_mod.Worker("127.0.0.1", 0, TOKEN, out, state_dir=state, name="w1", awake=False).start()
    try:
        c = WorkerClient(w.url, TOKEN)
        info = c.info()
        assert info["queue"]["paused"] and info["queue"]["pause"]["mode"] == "drain"
        time.sleep(0.5)
        assert c.status(job)["state"] == "queued"
        # `worker pause` from the command line writes the same file; the worker notices it
        control.write_pause(state, False, by="cli")
        wait_for(lambda: not c.info()["queue"]["paused"], timeout=15, what="the file's resume")
        assert c.wait(job, poll=0.1, timeout=120)["state"] == "done"
    finally:
        w.stop(10)


def test_pause_endpoint_and_sync_validation(make_worker):
    import urllib.error
    import urllib.request

    w = make_worker("w1")

    def post(path, body):
        req = urllib.request.Request(w.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                                              worker_mod.VERSION_HEADER: VERSION})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    for body in ({"paused": "yes"}, {"paused": True, "mode": "later"}, {"paused": True, "x": 1},
                 {"paused": True, "by": "a\nb"}):
        assert post("/v1/pause", body)[0] == 400, body
    assert post("/v1/pause", {"paused": True})[1]["paused"] is True       # the old body still works
    job = WorkerClient(w.url, TOKEN).submit(inputs(2), dict(CONFIG), "v")
    for body in ({"revoke": "ethanol"}, {"revoke": [1]}, {"following": "no"}, {"x": []}):
        assert post(f"/v1/jobs/{job}/sync", body)[0] == 400, body
    assert post("/v1/jobs/0123456789ab/sync", {"revoke": []})[0] == 404
    code, got = post(f"/v1/jobs/{job}/sync", {"revoke": ["ethanol"], "following": False})
    assert code == 200 and got["moved"] == ["ethanol"] and got["job"]["counts"].get("moved") == 1


def test_wake_holds_followed_jobs_until_the_client_syncs(make_worker, monkeypatch):
    started = stoppable(monkeypatch, 0.3)
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    job = c.submit(inputs(8), dict(CONFIG), "wk")
    wait_for(lambda: started, what="a start")
    for _ in range(3):
        c.status(job, follow=True)                                   # a grid client follows it
    held = w.manager.on_wake(300.0)
    assert held == [job]
    st = c.status(job, follow=True)
    assert st["rejoin"] and c.info()["queue"]["rejoin"] == [job]
    time.sleep(0.8)
    n = len(started)
    time.sleep(0.8)
    assert len(started) == n                                            # holds: no new molecule
    pending = [i for i, s in c.molecules(job).items() if s == "pending"]
    got = c.sync(job, pending[:2], following=True)                    # the client gave 2 to another machine
    assert sorted(got["moved"]) == sorted(pending[:2]) and not c.status(job)["rejoin"]
    st = c.wait(job, poll=0.1, timeout=120)
    assert st["state"] == "done" and st["counts"]["moved"] == 2


def test_lost_contact_and_expiry(tmp_path, monkeypatch):
    m = worker_mod.JobManager(tmp_path / "runs", "w1", state_dir=tmp_path / "state", awake=False, watch=False)
    m.paused = True
    m.start()
    try:
        job = m.submit(worker_mod.validate_job(client_mod.job_body(inputs(2), dict(CONFIG), "lc"), VERSION))
        for _ in range(3):
            m.note_poll(job, True)
        m.away_after_s, m.rejoin_wait_s = 0.2, 0.5
        time.sleep(0.3)
        m._check_contact()
        assert job.rejoin is not None and "no word" in job.rejoin["why"]
        time.sleep(0.6)
        m._check_contact()
        assert job.rejoin is None and not job.following                     # waited long enough: goes on
        m.note_poll(job, True)
        m.on_wake(100.0)
        assert job.rejoin is not None
        m.sync(job.id, [], following=False)
        assert job.rejoin is None and not job.following
    finally:
        m.stop(5)


def test_pause_on_battery_and_back_on_power(tmp_path, monkeypatch):
    m = worker_mod.JobManager(tmp_path / "runs", "lap", state_dir=tmp_path / "state", awake=False, watch=False)
    m.start()
    power = {"on_battery": True}
    monkeypatch.setattr(control, "battery", lambda: {"battery": True, "percent": 80.0, **power})
    try:
        assert settings_mod.load().machine_control_for("lap")["pause_on_battery"] is True     # the default
        m._read_settings()
        m._check_battery()
        assert m.paused and m.pause_info["reason"] == "battery" and control.read_pause(m.pause_dir)["paused"]
        power["on_battery"] = False
        m._check_battery()
        assert not m.paused
        # a pause by hand is not lifted by the power coming back
        m.set_paused(True, by="me")
        m._check_battery()
        assert m.paused
        m.set_paused(False)
        settings_mod.update_machine("lap", pause_on_battery=False)
        m._read_settings()
        power["on_battery"] = True
        m._check_battery()
        assert not m.paused
    finally:
        m.stop(5)


def test_clock_watch_sees_a_sleep():
    mono, wall = [0.0], [1000.0]
    cw = control.ClockWatch(interval=5.0, gap=30.0, clock=lambda: mono[0], wall=lambda: wall[0])
    mono[0] += 5
    wall[0] += 5
    assert cw.check() == 0.0
    mono[0] += 5                       # macOS / Linux: monotonic stands still during the sleep
    wall[0] += 605
    assert cw.check() == pytest.approx(600.0) and cw.last_wake and cw.asleep_from == 1005.0
    mono[0] += 400                     # a monotonic clock that counts the sleep: the check is late
    wall[0] += 400
    assert cw.check() == pytest.approx(395.0)
    mono[0] += 20
    wall[0] += 20                      # a busy moment, under the gap
    assert cw.check() == 0.0


# ------------------------------------------------------------------------------------- MachineWatch
def _grid(tmp_path, shares, weights=None):
    return client_mod.GridRun(id="g", folder=str(tmp_path), label="t", created="", client="me",
                              quantabench_version="x", method="mmff", config={}, weights=weights or {}, ids=[],
                              shares=shares)


def test_machine_watch_paused_away_back(tmp_path):
    clock = [0.0]
    a = Share(machine="mac", url="http://100.1.1.1:1", n=6, job_id="a" * 12, state="running",
              ids=[f"m{k}" for k in range(6)], counts={"ok": 1, "running": 1, "pending": 4})
    b = Share(machine="nova", url="http://100.1.1.2:1", n=6, job_id="b" * 12, state="running",
              ids=[f"n{k}" for k in range(6)], counts={"ok": 1, "pending": 5})
    grid = _grid(tmp_path, [a, b], {"mac": 1.0, "nova": 1.0})
    calls: list = []
    snap = {"m0": "ok", "m1": "running", "m2": "pending", "m3": "pending", "m4": "pending", "m5": "pending"}

    def release(share, n, dest):
        calls.append(("release", share.machine, n))
        return ["m2", "m3", "m4", "m5"][:n]

    def submit(machine, ids, parent, why="work sharing"):
        calls.append(("submit", machine, tuple(ids), why))
        s = Share(machine=machine, url="x", n=len(ids), job_id=f"{len(grid.shares):012x}", state="queued",
                  parent=parent.job_id, ids=list(ids), reason=why)
        grid.shares.append(s)
        return s

    def sync(share, revoke):
        calls.append(("sync", share.machine, tuple(revoke)))
        return {"moved": [i for i in revoke if i != "m2"], "done": ["m2"], "running": []}

    w = MachineWatch(grid, release, submit, lambda s: dict(snap) if s.machine == "mac" else None, sync,
                     log=calls.append, clock=lambda: clock[0], away_after_s=120, away_running_after_s=600)
    # paused: its 4 unstarted molecules go to nova
    w.ok(a, {"state": "running", "machine_paused": True})
    w.ok(b, {"state": "running"})
    w.step()
    assert ("release", "mac", 4) in calls and ("submit", "nova", ("m2", "m3", "m4", "m5"), "paused") in calls
    assert a.n == 2 and a.ids == ["m0", "m1"] and grid.moves[-1] == {"from": "mac", "to": "nova", "n": 4,
                                                                      "why": "paused"}
    # resumed, then the lid closes: no answer for 2 minutes -> away; its unstarted molecules go to nova
    a.ids, a.n = [f"m{k}" for k in range(6)], 6
    grid.shares = [a, b]
    calls.clear()
    w.ok(a, {"state": "running"})
    clock[0] = 10.0
    w.failed(a, client_mod.WorkerError("timed out"))
    w.ok(b, {"state": "running"})
    clock[0] = 140.0
    w.ok(b, {"state": "running"})
    w.step()
    assert "mac" in w.away and a.away_since
    sub = [c for c in calls if isinstance(c, tuple) and c[0] == "submit"]
    assert sub == [("submit", "nova", ("m2", "m3", "m4", "m5"), "away")] and a.reassigned == ["m2", "m3", "m4", "m5"]
    assert not w.settled(a)                              # m1 was running there: wait for the longer grace
    clock[0] = 700.0
    w.ok(b, {"state": "running"})                        # nova keeps answering (follow polls every few seconds)
    w.step()
    assert a.reassigned[-1] == "m1" and w.settled(a)     # in-progress ones too; nothing left to wait for
    # back: the client tells it which ones moved; m2 had been finished there too (merged once)
    w.ok(a, {"state": "running"})
    assert ("sync", "mac", ("m2", "m3", "m4", "m5", "m1")) in calls and "mac" not in w.away and not a.away_since
    assert "m3" not in a.ids and a.n == 6 - 4


def test_machine_watch_gives_up_without_receivers(tmp_path):
    clock = [0.0]
    a = Share(machine="mac", url="http://100.1.1.1:1", n=3, job_id="a" * 12, state="running", ids=["x", "y", "z"])
    grid = _grid(tmp_path, [a])
    notes: list = []
    w = MachineWatch(grid, lambda *a: [], lambda *a: None, lambda s: None, lambda s, r: {}, log=notes.append,
                     clock=lambda: clock[0], away_after_s=60, away_running_after_s=120)
    w.failed(a, None)
    clock[0] = 61.0
    w.step()
    assert "mac" in w.away and not w.settled(a)
    clock[0] = 200.0
    assert w.settled(a) and any("not waiting for it" in n for n in notes)


# ------------------------------------------------------------------------------------- end to end
def _items():
    return [from_smiles(s, id=i) for i, s in MOLS]


def _grid_cfg(out):
    return pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                                formats=("xyz",), output_dir=str(out), rank_method="refine")


def test_grid_moves_a_paused_machines_molecules(make_worker, qb_settings, monkeypatch):
    slowed = stoppable(monkeypatch, 0.35)
    out = Path(qb_settings.output_dir)
    w1, w2 = make_worker("w1", output=out), make_worker("w2", output=out)
    logs: list[str] = []
    done: dict = {}

    def run():
        done["grid"] = client_mod.grid_build(_items(), _grid_cfg(out), ["w1", "w2"], "1,1", label="pz", wait=True,
                                             extra_workers=[f"w1={w1.url}", f"w2={w2.url}"], poll=0.2, sync_wait=0,
                                             log=logs.append, share_work=False)

    th = threading.Thread(target=run)
    th.start()
    wait_for(lambda: len(slowed) >= 2, what="the grid to start")
    WorkerClient(w1.url, TOKEN).pause(True, "drain", by="test")
    th.join(240)
    grid = done["grid"]
    merged = summary_ids(grid.folder)
    assert sorted(merged) == sorted(i for i, _ in MOLS) and len(merged) == len(set(merged)), logs
    assert grid.missing == [] and any(m["why"] == "paused" and m["from"] == "w1" for m in grid.moves), logs
    assert grid.by_machine.get("w2", 0) > grid.by_machine.get("w1", 0)
    assert any(line.startswith("Moved:") and "(paused)" in line for line in logs)


def test_grid_away_machine_reassigned_deduped_and_wait_finishes(make_worker, qb_settings, monkeypatch):
    """w1 keeps computing but stops answering (its network is gone): after away_after_s its unstarted molecules go
    to w2, later the running ones too; --wait finishes; when w1 answers again the client syncs; molecules done on
    both are merged once (first done) and the timings count them once."""
    ids = [i for i, _ in MOLS]
    stoppable(monkeypatch, 0.5)
    out = Path(qb_settings.output_dir)
    w1, w2 = make_worker("w1", output=out), make_worker("w2", output=out)
    dark = {"on": False}
    real_status = WorkerClient.status

    def status(self, job_id, tail=20, follow=False):
        if dark["on"] and self.url == w1.url:
            raise client_mod.WorkerError(f"cannot reach {self.url}: timed out")
        return real_status(self, job_id, tail, follow)

    monkeypatch.setattr(WorkerClient, "status", status)
    logs: list[str] = []
    done: dict = {}

    def run():
        done["grid"] = client_mod.grid_build(_items(), _grid_cfg(out), ["w1", "w2"], "3,1", label="aw", wait=True,
                                             extra_workers=[f"w1={w1.url}", f"w2={w2.url}"], poll=0.2, sync_wait=0,
                                             log=logs.append, share_work=False, away_after_s=1.0,
                                             away_running_after_s=2.5)

    th = threading.Thread(target=run)
    th.start()
    wait_for(lambda: any(line.startswith("w1:") and "molecule(s) -> job" in line for line in logs), what="submit")
    time.sleep(1.0)
    dark["on"] = True                                       # the lid closes (here: the network goes)
    wait_for(lambda: any("is away" in line or "it is away" in line for line in logs), timeout=60, what="away")
    th.join(240)
    assert not th.is_alive()
    grid = done["grid"]
    merged = summary_ids(grid.folder)
    assert sorted(merged) == sorted(ids) and len(merged) == len(set(merged)), logs
    w1_share = grid.shares[0]
    assert w1_share.machine == "w1" and w1_share.reassigned, logs
    assert any(m["why"].startswith("away") for m in grid.moves)
    # every molecule once; those w1 also finished are recorded as duplicates, kept from the first to finish
    kept = {d["id"]: d["kept"] for d in grid.duplicates}
    assert kept, logs                       # w1 went on computing in the dark: those it finished twice are merged once
    for d in grid.duplicates:
        assert len(d["dropped"]) == 1 and d["kept"] != d["dropped"][0]
    assert sum(grid.by_machine.values()) == len(ids)
    # the timings count each molecule once (the dropped result is marked)
    from quantabench.dist import timings

    folders = timings.find_job_tables([out])
    got = timings.read_rows(folders)
    timings.mark_duplicates(got.rows, timings.grid_duplicates([out]))
    counts = timings.molecules_by_machine(got.rows)
    counted = sum(c["ok"] + c["failed"] for c in counts.values())
    assert counted == len(ids) and sum(c["duplicates"] for c in counts.values()) == len(kept)
    # w1 answers again: the client (a later `grid status`/new follow) syncs; its worker drops what moved
    dark["on"] = False
    job = w1_share.job_id
    got = WorkerClient(w1.url, TOKEN).sync(job, w1_share.reassigned, following=False)
    assert set(got["moved"]) | set(got["done"]) | set(got["running"]) <= set(w1_share.reassigned)
    st = WorkerClient(w1.url, TOKEN).wait(job, poll=0.1, timeout=120)
    assert st["state"] == "done"


# ------------------------------------------------------------------------------------- the split
def test_weights_equal_measured_auto_and_exclude(tmp_path, qb_settings, monkeypatch):
    from quantabench.dist import profile

    profs = [profile.MachineProfile(name="nova", logical_cores=24, physical_cores=12,
                                    throughput={"mmff@cpu": 3000.0}),
             profile.MachineProfile(name="mac", logical_cores=14, physical_cores=14, throughput={"mmff@cpu": 1000.0})]
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: list(profs))
    names = ["nova", "mac"]
    assert client_mod.resolve_weights("equal", names, "mmff", {}) == {"nova": 1.0, "mac": 1.0}
    assert client_mod.resolve_weights("benchmark", names, "mmff", {}) == {"nova": 3000.0, "mac": 1000.0}
    # measured: the last grid of the same protocol wins over the benchmark
    cfg = client_mod.config_for_workers(cfg_of())
    g = tmp_path / "out" / "2026-11-11_120000_grid_x"
    g.mkdir(parents=True)
    (g / "grid.json").write_text(json.dumps({"id": g.name, "client": "nova", "config": cfg,
                                             "rates": {"local": 400.0, "mac": 800.0}}), encoding="utf-8")
    monkeypatch.setattr(profile, "machine_name", lambda: "nova")
    got = client_mod.resolve_weights("measured", names, "mmff", {}, config=cfg, output_dir=tmp_path / "out")
    assert got == {"nova": 400.0, "mac": 800.0}
    other = {**cfg, "max_conformers": 50}                     # another protocol: the benchmark
    assert client_mod.resolve_weights("measured", names, "mmff", {}, config=other,
                                      output_dir=tmp_path / "out") == {"nova": 3000.0, "mac": 1000.0}
    # auto: each machine's split setting; manual on the scale where 1 is the mean of the Auto machines
    settings_mod.update_machine("mac", split="manual", weight=2.0)
    got = client_mod.resolve_weights("auto", names, "mmff", {})
    assert got == {"nova": 1.0, "mac": 2.0}
    settings_mod.update_machine("mac", split="off")
    assert client_mod.resolve_weights("auto", names, "mmff", {})["mac"] == 0.0
    settings_mod.update_machine("nova", split="off")
    with pytest.raises(client_mod.GridError, match="split is off"):
        client_mod.resolve_weights("auto", names, "mmff", {})


def test_grid_exclude_and_split_off_leave_machines_out(make_worker, qb_settings, monkeypatch):
    out = Path(qb_settings.output_dir)
    w1 = make_worker("w1", output=out)
    logs: list[str] = []
    grid = client_mod.grid_build(_items()[:3], _grid_cfg(out), ["w1", "local"], "1,1", label="ex", wait=True,
                                 extra_workers=[f"w1={w1.url}"], poll=0.2, sync_wait=0, log=logs.append,
                                 exclude=["local"])
    assert [s.machine for s in grid.shares] == ["w1"] and grid.excluded == ["local"] and grid.missing == []
    settings_mod.update_machine("w1", split="off")
    with pytest.raises(client_mod.GridError, match="left out"):
        client_mod.grid_build(_items()[:3], _grid_cfg(out), ["w1"], "auto", label="ex2", wait=True,
                              extra_workers=[f"w1={w1.url}"], poll=0.2, sync_wait=0, log=logs.append)


# ------------------------------------------------------------------------------------- settings
def test_settings_merge_keeps_every_machines_change(isolated_settings):
    a = settings_mod.load()                      # the GUI on one machine, open for a while
    settings_mod.update_machine("nova", priority="background", free_threads=6, lanes="auto")
    settings_mod.update_machine("mac", split="manual", weight=2.5, pause_on_battery=True)
    a.build.max_conformers = 77                  # a's own change of a shared setting
    settings_mod.save(a)
    s = settings_mod.load()
    assert s.priority_for("nova") == "background" and s.free_threads_for("nova") == 6 and s.lanes_for("nova") == "auto"
    c = s.machine_control_for("mac")
    assert (c["split"], c["weight"], c["pause_on_battery"]) == ("manual", 2.5, True)
    assert s.build.max_conformers == 77 and "resources.machine_priority" in s.dist.machine_updated["nova"]
    # another process changed a shared value meanwhile: a save that did not touch it keeps it
    b = settings_mod.load()
    other = settings_mod.load()
    other.build.energy_window_kcal = 9.0
    settings_mod.save(other)
    b.build.ph = 3.0
    settings_mod.save(b)
    s = settings_mod.load()
    assert s.build.energy_window_kcal == 9.0 and s.build.ph == 3.0


def test_saving_unchanged_settings_does_not_write(isolated_settings):
    path = settings_mod.settings_dir() / settings_mod.SETTINGS_FILE
    settings_mod.update_machine("nova", priority="low")
    before = path.stat().st_mtime_ns
    time.sleep(0.02)
    s = settings_mod.load()
    settings_mod.save(s)                         # the GUI saves at every Run: nothing changed, nothing written
    assert path.stat().st_mtime_ns == before


def test_conflict_copy_with_a_newer_entry_is_merged_back(isolated_settings):
    folder = settings_mod.settings_dir()
    settings_mod.update_machine("nova", priority="low")
    main = json.loads((folder / settings_mod.SETTINGS_FILE).read_text(encoding="utf-8"))
    # the Mac changed its own lanes at the same time; SyncThing kept NOVA's file and renamed the Mac's
    theirs = json.loads(json.dumps(main))
    theirs["dist"]["machine_lanes"] = {"mac": "6x2"}
    theirs["dist"]["machine_updated"]["mac"] = {"dist.machine_lanes": "2999-01-01T00:00:00+00:00"}
    theirs["resources"]["machine_priority"] = {"nova": "normal"}         # older for nova: not taken
    (folder / "settings.sync-conflict-20261111-101500-ABCDEFG.json").write_text(json.dumps(theirs), encoding="utf-8")
    s = settings_mod.load()
    assert s.lanes_for("mac") == "6x2" and s.priority_for("nova") == "low"
    assert s.recovered_conflicts and "mac dist.machine_lanes" in s.recovered_conflicts[0]
    settings_mod.save(s)
    disk = json.loads((folder / settings_mod.SETTINGS_FILE).read_text(encoding="utf-8"))
    assert disk["dist"]["machine_lanes"] == {"mac": "6x2"}
    assert settings_mod.load().recovered_conflicts == []                 # merged: nothing newer any more


def test_machines_set_cli_and_max_cpu(isolated_settings, capsys, monkeypatch):
    from quantabench import cli
    from quantabench.dist import profile

    monkeypatch.setattr(profile, "load_profile", lambda name: profile.MachineProfile(name=name, logical_cores=24)
                        if name == "btm-nova" else None)
    assert cli.main(["machines", "set", "btm-nova", "--priority", "background", "--free-threads", "6",
                     "--lanes", "auto"]) == 0
    s = settings_mod.load()
    assert (s.priority_for("btm-nova"), s.free_threads_for("btm-nova"), s.lanes_for("btm-nova")) == \
        ("background", 6, "auto")
    assert cli.main(["machines", "set", "btm-nova", "--max-cpu", "50"]) == 0
    assert settings_mod.load().free_threads_for("btm-nova") == 12
    assert control.free_threads_for_max_cpu(100, 24) == 0 and control.free_threads_for_max_cpu(1, 24) == 23
    assert cli.main(["machines", "set", "btm-nova", "--weight", "2", "--pause-on-battery"]) == 0
    c = settings_mod.load().machine_control_for("btm-nova")
    assert (c["split"], c["weight"], c["pause_on_battery"]) == ("manual", 2.0, True)
    assert cli.main(["machines", "set", "btm-nova", "--lanes", "gpu:x"]) == 2
    assert cli.main(["machines", "set", "elsewhere", "--max-cpu", "50"]) == 2       # no thread count known
    assert cli.main(["machines", "set", "btm-nova", "--priority", "default", "--free-threads", "default"]) == 0
    s = settings_mod.load()
    assert s.priority_for("btm-nova", scope=True)[1] == "all machines"
    capsys.readouterr()
    assert cli.main(["machines", "set", "btm-nova"]) == 0
    assert "split manual weight 2" in capsys.readouterr().out


def test_worker_pause_cli_this_machine_and_by_url(make_worker, capsys):
    from quantabench import cli

    assert cli.main(["worker", "pause", "--now"]) == 0
    assert control.read_pause()["paused"] and control.read_pause()["mode"] == "now"
    assert cli.main(["worker", "resume"]) == 0
    assert not control.read_pause()["paused"]
    w = make_worker("w1")
    assert cli.main(["worker", "pause", "--worker", f"w1={w.url}"]) == 0
    assert w.manager.paused and "w1 worker" in capsys.readouterr().out
    assert cli.main(["worker", "resume", "w1", "--worker", f"w1={w.url}"]) == 0
    assert not w.manager.paused
    assert cli.main(["worker", "pause", "nobody"]) == 2


def test_priority_applied_to_running_processes():
    import subprocess
    import sys

    from quantabench import priority

    if sys.platform == "win32":
        pytest.skip("POSIX nice values")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        note = priority.apply_to_pids([proc.pid], "low")
        import psutil

        assert psutil.Process(proc.pid).nice() == priority.NICE["low"] and "1 process(es) changed" in note
        note = priority.apply_to_pids([proc.pid], "normal")       # raising: not possible here without privileges
        assert "cannot be raised" in note or psutil.Process(proc.pid).nice() == 0
        assert "no running lane" in priority.apply_to_pids([], "low")
    finally:
        proc.kill()
        proc.wait(10)


def test_status_file_of_a_worker(make_worker):
    w = make_worker("w1", publish=True)
    WorkerClient(w.url, TOKEN).pause(True, "now")
    st = control.load_statuses()["w1"]
    assert st["paused"] and st["pause"]["mode"] == "now" and st["url"] == w.url
    # the registry never reads it as a machine profile
    from quantabench.dist import profile

    assert all(p.name != "w1.json" for p in profile.load_profiles())
