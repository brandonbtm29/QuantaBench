"""Work sharing between machines (``jobs.Handoff``, ``JobManager.release``, ``client.WorkSharing``): a batch hands
molecules it has not started to another machine, and a grid moves them from a slow machine to an idle one."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from quantabench import settings as settings_mod
from quantabench.chem import jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import client as client_mod
from quantabench.dist import plan, speed
from quantabench.dist import worker as worker_mod
from quantabench.dist.client import Share, WorkerClient, WorkSharing

TOKEN = "test-token-" + "y" * 32
MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F"),
        ("methanol", "CO"), ("acetone", "CC(C)=O"), ("propanol", "CCCO"), ("butane", "CCCC"),
        ("chloroform", "ClC(Cl)Cl"), ("dmso", "CS(C)=O"), ("acetonitrile", "CC#N"), ("formic_acid", "OC=O")]
CONFIG = {"method": "mmff", "max_conformers": 2, "refine_top": 1, "xtb_properties": False, "formats": ["xyz"]}


def inputs(n=None):
    return [from_smiles(s, id=i) for i, s in MOLS[:n]]


def wait_for(cond, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def summary_ids(folder):
    import csv

    with open(Path(folder) / "summary.csv", newline="", encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f)]


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


def slowed(monkeypatch, seconds, ids=None):
    """Make pipeline.run_molecule sleep ``seconds`` first (only for ``ids`` when given); returns the ids started."""
    real = pipeline.run_molecule
    started: list[str] = []

    def run(inp, *a, **k):
        started.append(inp.id)
        if ids is None or inp.id in ids:
            time.sleep(seconds)
        return real(inp, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", run)
    return started


# ------------------------------------------------------------------------------------------- Handoff
def test_handoff_request_serve_withdraw_close():
    h = jobs.Handoff()
    got: dict = {}
    t = threading.Thread(target=lambda: got.setdefault("ids", h.request(2, "w2", timeout=10)))
    t.start()
    wait_for(lambda: h._pending is not None)
    h.serve(lambda n, dest: [f"m{k}" for k in range(n)] if dest == "w2" else [])
    t.join(10)
    assert got["ids"] == ["m0", "m1"]
    # nobody serves in time: the request is withdrawn, and a late serve moves nothing
    taken: list = []
    assert h.request(3, timeout=0.05) == []
    h.serve(lambda n, d: taken.append(n) or ["x"])
    assert taken == [] and h.request(0) == []
    # close answers a waiting request at once, and every later one
    t = threading.Thread(target=lambda: got.__setitem__("late", h.request(1, timeout=30)))
    t0 = time.monotonic()
    t.start()
    wait_for(lambda: h._pending is not None)
    h.close()
    t.join(10)
    assert got["late"] == [] and time.monotonic() - t0 < 10 and h.closed and h.request(1, timeout=5) == []


def test_sequential_batch_hands_over_unstarted_molecules(tmp_path, monkeypatch):
    started: list[str] = []

    def run(inp, *a, **k):
        started.append(inp.id)
        time.sleep(0.25)
        return pipeline.MoleculeResult(input=inp, status="ok", method="mmff")

    monkeypatch.setattr(pipeline, "run_molecule", run)
    mols, run_dir, h = inputs(6), tmp_path / "run", jobs.Handoff()
    cfg = pipeline.BuildConfig(method="mmff", write_files=False)
    th = threading.Thread(target=lambda: jobs.run_resumable(mols, cfg, run_dir, handoff=h))
    th.start()
    wait_for(lambda: started)
    moved = h.request(2, "w2", timeout=10)
    th.join(60)
    assert moved == [mols[-1].id, mols[-2].id] and not set(moved) & set(started)   # from the end, not started
    st = jobs.open_run(run_dir)
    assert (st.counts["moved"], st.total, st.finished, st.remaining) == (2, 4, 4, 0)
    assert "2 moved to another machine" in st.summary() and h.closed
    with jobs.JobStore(run_dir) as store:
        assert store.mark_moved([started[0]], "w3") == []            # finished rows never move
        errors = dict(store.conn.execute("SELECT id, error FROM molecules WHERE status = 'moved'").fetchall())
    assert errors == {i: "moved to w2" for i in moved}
    n = len(started)
    jobs.run_resumable(None, cfg, run_dir)                            # a resume does not run them here
    assert len(started) == n


# ----------------------------------------------------------------------------------- worker endpoint
def test_worker_release_endpoint(make_worker, monkeypatch):
    import json
    import urllib.error
    import urllib.request

    slowed(monkeypatch, 0.4)
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    job = c.submit(inputs(6), dict(CONFIG), "rel")

    def post(path, body):
        req = urllib.request.Request(w.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                                              worker_mod.VERSION_HEADER: worker_mod.__version__})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    for body in ({"max": 0}, {"max": "2"}, {"max": True}, {"max": 2, "to": "a\nb"}, {"max": 2, "x": 1}):
        assert post(f"/v1/jobs/{job}/release", body)[0] == 400, body
    assert post("/v1/jobs/0123456789ab/release", {"max": 1})[0] == 404
    wait_for(lambda: c.status(job)["state"] == "running" and c.status(job)["counts"].get("ok", 0) >= 1)
    ids = c.release(job, 2, "w2")
    assert len(ids) == 2
    st = c.wait(job, poll=0.1, timeout=120)
    assert st["state"] == "done" and st["counts"]["moved"] == 2 and st["counts"]["ok"] == 4
    assert c.release(job, 1, "w2") == []                       # a finished job has nothing to hand over


# ------------------------------------------------------------------------------------------ balancer
def _grid(tmp_path, shares):
    return client_mod.GridRun(id="g", folder=str(tmp_path), label="t", created="", client="me",
                              quantabench_version="x", method="mmff", config={}, weights={}, ids=[], shares=shares)


def test_balancer_moves_the_share_that_evens_the_finish(tmp_path):
    clock = [0.0]
    fast = Share(machine="fast", n=10, job_id="a" * 12, state="running", counts={"ok": 0, "pending": 10})
    slow = Share(machine="slow", n=40, job_id="b" * 12, state="running", counts={"ok": 0, "pending": 40})
    grid = _grid(tmp_path, [fast, slow])
    released, notes = [], []

    def release(share, n, dest):
        released.append((share.machine, n, dest))
        return [f"m{k}" for k in range(n)]

    def submit(machine, ids, parent):
        s = Share(machine=machine, n=len(ids), job_id="c" * 12, state="queued", parent=parent.job_id, ids=list(ids))
        grid.shares.append(s)
        return s

    ws = WorkSharing(grid, release, submit, log=notes.append, clock=lambda: clock[0])
    assert ws.step() is None                                   # first sight: nothing measured yet
    clock[0] = 100.0
    fast.state, fast.counts = "done", {"ok": 10}
    slow.counts = {"ok": 10, "running": 1, "pending": 29}
    assert ws.rate("fast") == pytest.approx(0.1) and ws.rate("slow") == pytest.approx(0.1)
    # m = 29 * 0.1 / 0.2 = 14; slow alone 290 s, after: max(15/0.1, 30 + 14/0.1) = 170 s -> 120 s sooner
    new = ws.step()
    assert released == [("slow", 14, "fast")] and new is not None and new.parent == "b" * 12
    assert slow.n == 26 and ws.moves == [{"from": "slow", "to": "fast", "n": 14, "gain_s": 120.0}]
    assert ws.step() is None and len(released) == 1           # fast is busy again (its new share is queued)


def test_balancer_thresholds_failed_machines_and_failed_submits(tmp_path):
    clock = [0.0]

    def setup(fast_state="done", pending=29, min_gain_s=None, start_s=None):
        fast = Share(machine="fast", n=10, job_id="a" * 12, state="running")
        slow = Share(machine="slow", n=40, job_id="b" * 12, state="running")
        grid = _grid(tmp_path, [fast, slow])
        calls: list = []
        ws = WorkSharing(grid, lambda s, n, d: calls.append(("release", s.machine, n, d)) or [f"m{k}" for k in range(n)],
                         lambda m, ids, p: calls.append(("submit", m, len(ids))) or None, log=calls.append,
                         clock=lambda: clock[0], min_gain_s=min_gain_s, start_s=start_s)
        clock[0] = 0.0
        ws.step()
        clock[0] = 100.0
        fast.state, fast.counts = fast_state, {"ok": 10}
        slow.counts = {"ok": 10, "pending": pending}
        return ws, calls, slow

    ws, calls, _ = setup(pending=3)        # 3 left: moving 1 saves 10 s < 60 s
    assert ws.step() is None and calls == []
    ws, calls, _ = setup(pending=3, min_gain_s=5, start_s=0)      # 30 s -> max(20, 10): 10 s sooner
    assert ws.plan("fast", 100.0)[2] == 1
    for state in ("failed", "cancelled", "lost", "interrupted"):
        ws, calls, _ = setup(fast_state=state)
        assert ws.step() is None and calls == []
    # the thief cannot take them, nor can the donor: logged, left for grid.missing
    ws, calls, slow = setup()
    assert ws.step() is None
    assert [c for c in calls if isinstance(c, tuple)] == [("release", "slow", 14, "fast"), ("submit", "fast", 14),
                                                           ("submit", "slow", 14)]
    assert len(ws.unplaced) == 14 and any("no machine could start them" in c for c in calls if isinstance(c, str))


# ------------------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("machines", [("w1", "w2"), ("w1", "local")])
def test_grid_moves_unstarted_molecules_to_the_idle_machine(make_worker, qb_settings, monkeypatch, machines):
    ids = [i for i, _ in MOLS]
    slow = set(plan.split(ids, {machines[0]: 2.0, machines[1]: 1.0})[machines[0]])
    slowed(monkeypatch, 0.8, slow)
    out = Path(qb_settings.output_dir)
    w1 = make_worker("w1", output=out)
    extra = [f"w1={w1.url}"]
    if "w2" in machines:
        w2 = make_worker("w2")
        extra.append(f"w2={w2.url}")
    items = [from_smiles(s, id=i) for i, s in MOLS]
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                               formats=("xyz",), output_dir=str(out))
    logs: list[str] = []
    grid = client_mod.grid_build(items, cfg, list(machines), "2,1", label="ws", wait=True, extra_workers=extra,
                                 poll=0.2, sync_wait=0, log=logs.append, share_min_gain_s=0, share_start_s=0,
                                 share_min_done=1)
    merged = summary_ids(grid.folder)
    assert sorted(merged) == sorted(ids) and len(merged) == len(set(merged)), logs    # every molecule once
    assert grid.missing == [] and grid.state == "merged"
    moved = [s for s in grid.shares if s.parent]
    assert moved, logs
    by_job = {s.job_id: s for s in grid.shares}
    for s in moved:
        assert s.parent in by_job and by_job[s.parent].machine != s.machine and len(s.ids) == s.n
    assert sum(s.n for s in grid.shares) == len(ids) and sum(s.finished for s in grid.shares) == len(ids)
    first = grid.shares[0]
    assert first.machine == "w1" and first.counts.get("moved", 0) >= 1
    assert any(line.startswith("Work sharing:") for line in logs)
    # largest molecules first in w1's batch
    with sqlite3.connect(Path(first.run_dir) / jobs.DB_NAME) as con:
        order = [r[0] for r in con.execute("SELECT id FROM molecules ORDER BY seq")]
    size = {x.id: speed.molecule_size(x) for x in items}
    assert [size[i] for i in order] == sorted((size[i] for i in order), reverse=True)
    # several shares of one machine: separate download folders, and grid status lists every share
    reloaded = client_mod.GridRun.load(Path(grid.folder) / client_mod.GRID_FILE)
    assert len(reloaded.shares) == len(grid.shares) and any(s.parent for s in reloaded.shares)
