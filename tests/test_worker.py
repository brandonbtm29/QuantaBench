"""Workers and grid builds (quantabench.dist.worker / dist.client), all on 127.0.0.1 with threads.

Each worker gets its own port (0 = any free one), state folder and output folder; the settings folder is the
per-test one from conftest.py. Molecules are tiny and run with MMFF, without the xTB property step.
"""

import csv
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from quantabench import __version__, cli
from quantabench import settings as settings_mod
from quantabench.chem import pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import client as client_mod
from quantabench.dist import plan, profile
from quantabench.dist import worker as worker_mod
from quantabench.dist.client import WorkerClient, WorkerError

TOKEN = "test-token-" + "x" * 32
MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F"),
        ("methanol", "CO"), ("acetone", "CC(C)=O")]
CONFIG = {"method": "mmff", "max_conformers": 2, "refine_top": 1, "xtb_properties": False, "formats": ["xyz"]}
FAST = ["--method", "mmff", "--max-confs", "2", "--refine-top", "1", "--formats", "xyz"]


def inputs(names=None):
    return [from_smiles(s, id=i) for i, s in MOLS if names is None or i in names]


@pytest.fixture(autouse=True)
def qb_settings(isolated_settings, monkeypatch):
    s = settings_mod.Settings(output_dir=str(isolated_settings / "client_runs"))
    s.resources.threads, s.resources.gpu = 1, "off"
    s.dist.worker_token = TOKEN
    settings_mod.save(s)
    monkeypatch.setattr("quantabench.chem.properties.available", lambda: False)  # no dxtb step in tests
    return s


@pytest.fixture
def make_worker(tmp_path):
    started = []

    def make(name, output=None, state=None, **kw):
        w = worker_mod.Worker("127.0.0.1", 0, TOKEN, output or tmp_path / f"{name}_runs",
                              state_dir=state or tmp_path / f"{name}_state", name=name, awake=False, **kw)
        started.append(w)
        return w.start()

    yield make
    for w in started:
        w.stop(timeout=30)


def raw(url, method="GET", body=None, token=TOKEN, version=__version__):
    """A bare request (no WorkerClient), returning (status, json)."""
    data = body if isinstance(body, bytes) else None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if version:
        req.add_header(worker_mod.VERSION_HEADER, version)
    if data:
        req.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for(cond, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def summary_ids(folder):
    with open(Path(folder) / "summary.csv", newline="", encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f)]


# --------------------------------------------------------------------------------------- binding
ANY4, ANY6 = "0.0.0.0", "::"  # noqa: S104 (every network: what the worker must refuse)


def test_bind_rules(monkeypatch):
    with pytest.raises(ValueError, match="allow-any-address"):
        worker_mod.resolve_bind(ANY4)
    with pytest.raises(ValueError, match="allow-any-address"):
        worker_mod.resolve_bind(ANY6)
    assert worker_mod.resolve_bind(ANY4, allow_any=True) == ANY4
    assert worker_mod.resolve_bind("localhost") == "127.0.0.1"
    with pytest.raises(ValueError):
        worker_mod.resolve_bind("example.com")
    # `tailscale ip -4` (faked): the 100.x address is taken, anything else is ignored
    monkeypatch.setattr(worker_mod, "tailscale_executable", lambda: "tailscale")
    monkeypatch.setattr(worker_mod.subprocess, "run", lambda *a, **k: type(
        "P", (), {"stdout": "192.168.1.4\n100.90.226.110\n", "stderr": "", "returncode": 0})())
    assert worker_mod.resolve_bind("tailscale") == "100.90.226.110"
    # the CLI refuses before it generates a token or opens a socket
    assert cli.main(["worker", "--bind", ANY4, "--port", "0"]) == 2


def test_token_generated_once(qb_settings):
    s = settings_mod.load()
    s.dist.worker_token = ""
    settings_mod.save(s)
    token = worker_mod.ensure_token()
    assert len(token) >= worker_mod.MIN_TOKEN_LENGTH and settings_mod.load().dist.worker_token == token
    assert worker_mod.ensure_token() == token


# ------------------------------------------------------------------------------------ auth, info
def test_auth_info_and_discovery(make_worker, capsys):
    w1 = make_worker("w1", publish=True)       # writes its URL into machines/w1.json
    w2 = make_worker("w2")
    assert w1.port != w2.port
    for url in (w1.url + "/v1/info", w1.url + "/v1/jobs"):
        assert raw(url, token=None)[0] == 401
        assert raw(url, token="wrong-" + TOKEN)[0] == 401
    assert raw(w1.url + "/v1/jobs", "POST", {"inputs": []}, token=None)[0] == 401
    status, info = raw(w1.url + "/v1/info")
    assert status == 200 and info["name"] == "w1" and info["api"] == worker_mod.API_VERSION
    assert info["quantabench_version"] == __version__ and info["profile"]["address"] == w1.url
    assert info["queue"] == {"running": [], "queued": [], "paused": False, "max_jobs": 1, "hold": ""}
    assert TOKEN not in json.dumps(info)
    assert raw(w1.url + "/v1/nothing")[0] == 404
    assert raw(w1.url + "/v1/info", "POST", {})[0] == 405
    # discovery: the registry (w1 published itself), settings overrides, --worker
    s = settings_mod.load()
    s.dist.workers = {"w2": w2.url}
    settings_mod.save(s)
    refs = client_mod.discover()
    assert refs["w1"].source == "registry" and refs["w1"].reachable
    assert refs["w2"].source == "settings" and refs["w2"].info["name"] == "w2"
    assert cli.main(["grid", "machines"]) == 0
    out = capsys.readouterr().out
    assert "w1" in out and "w2" in out and w1.url in out
    # plain HTTP only to loopback/Tailscale addresses (the token would cross the LAN in clear text)
    with pytest.raises(WorkerError, match="not a Tailscale"):
        WorkerClient("http://192.168.1.5:8765", TOKEN).info()


# ------------------------------------------------------------------------------------ validation
def test_validation_and_version(make_worker):
    w = make_worker("w1", max_request_bytes=200_000)
    good = client_mod.job_body(inputs(["ethanol"]), dict(CONFIG), "t1")

    def cfg(**kw):
        return {**good, "config": {**good["config"], **kw}}

    cases = [
        (cfg(bogus=1), "unknown config field"),
        (cfg(output_dir="~/elsewhere"), "output_dir"),
        (cfg(formats=["xyz", "exe"]), "formats"),
        (cfg(method="rm -rf /"), "method"),
        (cfg(max_conformers="5"), "integer"),
        (cfg(ph=99.0), "ph"),
        (cfg(qm_method="b3lyp/def2-svp\nimport os"), "qm_method"),
        ({**good, "label": "../../evil"}, "label"),
        ({**good, "inputs": [json.dumps({"id": "../evil", "smiles": "CCO"})]}, "folder-safe"),
        ({**good, "inputs": [json.dumps({"id": "a/b", "smiles": "CCO"})]}, "folder-safe"),
        ({**good, "inputs": [json.dumps({"id": "CON", "smiles": "CCO"})]}, "reserved"),
        ({**good, "inputs": [json.dumps({"id": "x", "smiles": "CCO", "evil": 1})]}, "unknown field"),
        ({**good, "inputs": good["inputs"] * 2}, "duplicate"),
        ({**good, "chunk": {"index": 0, "count": 2, "weights": [1, 1]}}, "chunk"),
        ({**good, "lanes": "; rm -rf"}, "lanes"),
        ({**good, "extra": 1}, "unknown field"),
    ]
    for body, message in cases:
        status, data = raw(w.url + "/v1/jobs", "POST", body)
        assert status == 400 and message in data["error"], (body, data)
    assert raw(w.url + "/v1/jobs", "POST", b"{not json")[0] == 400
    big = {**good, "inputs": [json.dumps({"id": f"m{i}", "smiles": "C" * 100}) for i in range(2000)]}
    assert raw(w.url + "/v1/jobs", "POST", big)[0] == 413
    # a different QuantaBench version is refused, unless the job allows it
    status, data = raw(w.url + "/v1/jobs", "POST", good, version="0.0.1")
    assert status == 409 and "version" in data["error"]
    assert raw(w.url + "/v1/jobs", "POST", good, version=None)[0] == 409
    assert w.manager.list() == [] and not w.manager.output_dir.exists()
    status, data = raw(w.url + "/v1/jobs", "POST", {**good, "allow_version_mismatch": True}, version="0.0.1")
    assert status == 201 and data["job"]["client_version"] == "0.0.1"


# ------------------------------------------------------------------------- submit, cancel, archive
def test_submit_status_cancel_archive(make_worker, monkeypatch, tmp_path):
    gate = threading.Event()
    real = pipeline.run_molecule

    def held(inp, cfg, device, threads, should_stop, *a, **k):
        while not gate.is_set() and not should_stop():
            time.sleep(0.02)
        return real(inp, cfg, device, threads, should_stop, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", held)
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    a = c.submit(inputs(["ethanol", "TFE"]), dict(CONFIG), "jobA")
    b = c.submit(inputs(["methanol"]), dict(CONFIG), "jobB")
    wait_for(lambda: c.status(a)["state"] == "running")
    assert c.status(b)["state"] == "queued"
    with pytest.raises(WorkerError) as e:
        c.download_archive(a, tmp_path / "dl")
    assert e.value.status == 409
    assert c.cancel(b)["state"] == "cancelled"
    assert c.cancel(a)["cancel_requested"] is True
    st = c.wait(a, poll=0.1, timeout=60)
    assert st["state"] == "cancelled" and st["counts"]["ok"] == 0 and st["counts"]["pending"] >= 1
    assert {j["id"] for j in c.jobs()} == {a, b}
    with pytest.raises(WorkerError) as e:
        c.status("0123456789ab")
    assert e.value.status == 404

    gate.set()
    j = c.submit(inputs(["ethanol", "TFE"]), dict(CONFIG), "jobC")
    st = c.wait(j, poll=0.1, timeout=120)
    assert st["state"] == "done" and st["counts"]["ok"] == 2, st
    run_dir = Path(st["run_dir"])
    assert run_dir.parent == w.manager.output_dir.resolve() and st["run_folder"].endswith("_jobC_w1")
    assert (run_dir / worker_mod.JOB_FILE).is_file() and (run_dir / worker_mod.JOB_LOG).is_file()
    assert any("Job " + j in line for line in c.status(j, tail=100)["log_tail"])

    dest = c.download_archive(j, tmp_path / "dl")
    assert dest == tmp_path / "dl" / st["run_folder"]
    assert sorted(summary_ids(dest)) == ["TFE", "ethanol"] and list((dest / "ethanol").glob("*.xyz"))
    assert not (dest / "jobs.lock").exists()


def test_safe_extract_refuses_escapes(tmp_path):
    for name in ("../evil.txt", "/abs.txt", "run/../../evil.txt", "C:/evil.txt"):
        zpath = tmp_path / "a.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr(name, "x")
        with pytest.raises(WorkerError):
            client_mod.safe_extract(zpath, tmp_path / "out")
    zpath.write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(WorkerError, match="incomplete"):
        client_mod.safe_extract(zpath, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


# ----------------------------------------------------------------------------------------- restart
def test_restart_resumes(tmp_path, monkeypatch):
    first = threading.Event()
    calls = []
    real = pipeline.run_molecule

    def wrapped(inp, cfg, device, threads, should_stop, *a, **k):
        calls.append(inp.id)
        res = real(inp, cfg, device, threads, should_stop, *a, **k)
        if len(calls) == 1:            # hold the first molecule until the worker is told to stop
            first.set()
            wait_for(should_stop, timeout=30)
        return res

    monkeypatch.setattr(pipeline, "run_molecule", wrapped)
    state, out = tmp_path / "state", tmp_path / "runs"
    w = worker_mod.Worker("127.0.0.1", 0, TOKEN, out, state_dir=state, name="w1", awake=False).start()
    try:
        job_id = WorkerClient(w.url, TOKEN).submit(inputs(["ethanol", "TFE", "methanol"]), dict(CONFIG), "resume")
        assert first.wait(60)
    finally:
        w.stop(timeout=60)              # graceful: the job stops after this molecule and stays resumable
    data = json.loads((state / worker_mod.STATE_FILE).read_text(encoding="utf-8"))
    assert data["jobs"][0]["state"] == "interrupted"
    data["jobs"][0]["state"] = "running"   # and as if the worker had crashed mid-job
    (state / worker_mod.STATE_FILE).write_text(json.dumps(data), encoding="utf-8")

    w2 = worker_mod.Worker("127.0.0.1", 0, TOKEN, out, state_dir=state, name="w1", awake=False).start()
    try:
        st = WorkerClient(w2.url, TOKEN).wait(job_id, poll=0.1, timeout=120)
    finally:
        w2.stop(timeout=30)
    assert st["state"] == "done" and st["counts"]["ok"] == 3 and st["resumed"] == 1, st
    assert calls.count("ethanol") == 1 and sorted(calls) == ["TFE", "ethanol", "methanol"]
    assert sorted(summary_ids(st["run_dir"])) == ["TFE", "ethanol", "methanol"]
    assert not (state / "inputs" / f"{job_id}.jsonl").exists()


# -------------------------------------------------------------------------------------------- grid
def test_grid_build_two_workers(make_worker, qb_settings, capsys):
    client_out = Path(qb_settings.output_dir)
    w1 = make_worker("w1", output=client_out)   # its run folder appears here, as SyncThing would make it
    w2 = make_worker("w2")                       # a separate folder: its share comes back as an archive
    rc = cli.main(["grid", "build", *[f"{s} {i}" for i, s in MOLS], *FAST, "--machines", "w1,w2",
                   "--weights", "2,1", "--worker", f"w1={w1.url}", "--worker", f"w2={w2.url}", "--wait",
                   "--poll", "0.2", "--sync-wait", "0", "--label", "pfas"])
    assert rc == 0, capsys.readouterr()
    (grid_file,) = client_out.glob("*_grid_pfas/grid.json")
    grid = client_mod.GridRun.load(grid_file)
    ids = [i for i, _ in MOLS]
    expected = plan.split(ids, {"w1": 2.0, "w2": 1.0})
    merged = summary_ids(grid_file.parent)
    assert sorted(merged) == sorted(ids) and len(merged) == len(set(merged))   # every molecule once
    assert grid.missing == [] and grid.state == "merged" and grid.config["method"] == "mmff"
    for share in grid.shares:
        assert share.n == len(expected[share.machine])
        if share.n:
            assert share.state == "done" and share.finished == share.n
            assert share.source == {"w1": "synced", "w2": "archive"}[share.machine]
    assert "output_dir" not in grid.config and "threads" not in grid.config
    capsys.readouterr()
    assert cli.main(["grid", "status", grid.id]) == 0
    assert "6/6" in capsys.readouterr().out


def test_grid_build_with_local_share(make_worker, qb_settings, capsys):
    # the NOVA case: a worker on this very machine, writing into the same output folder, plus "local"
    here = profile.machine_name()
    client_out = Path(qb_settings.output_dir)
    w = make_worker(here, output=client_out)
    mols = [f"{s} {i}" for i, s in MOLS[:3]]
    # method 'auto' could differ per machine: refused before anything is submitted
    assert cli.main(["grid", "build", *mols, "--machines", f"local,{here}", "--worker", f"{here}={w.url}"]) == 2
    assert "explicitly" in capsys.readouterr().err
    assert cli.main(["grid", "build", *mols, *FAST, "--machines", "local,nova", "--worker",
                     f"{here}={w.url}"]) == 2
    assert "Unknown machine 'nova'" in capsys.readouterr().err
    items = inputs([i for i, _ in MOLS[:3]])
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                               formats=("xyz",), output_dir=qb_settings.output_dir)
    grid = client_mod.grid_build(items, cfg, ["local", here], "1,1", wait=True, poll=0.2, sync_wait=0,
                                 extra_workers=[f"{here}={w.url}"], log=lambda m: None)
    assert grid.missing == [] and sorted(summary_ids(grid.folder)) == sorted(x.id for x in items)
    local, remote = grid.shares
    assert local.n and remote.n and local.run_folder != remote.run_folder
    assert local.source == "local" and Path(local.run_dir).parent == client_out.resolve()
    assert local.run_folder.endswith(f"_{here}-local") and remote.source == "synced"
    assert sorted(summary_ids(local.run_dir) + summary_ids(client_out / remote.run_folder)) == \
        sorted(x.id for x in items)


# ======================================================================= review fixes (worker / grid)
def _grid(folder, shares, ids):
    folder.mkdir(parents=True, exist_ok=True)
    return client_mod.GridRun(id=folder.name, folder=str(folder), label="x", created="", client="c",
                              quantabench_version=__version__, method="mmff", config={}, weights={}, ids=ids,
                              shares=shares)


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_route_check_needs_tailscale_on_this_machine():
    check = client_mod.check_route
    check("127.0.0.1", "127.0.0.1")
    check("100.90.226.110", "100.101.7.8")                       # tailnet to tailnet
    check("fd7a:115c:a1e0::1", "fd7a:115c:a1e0::2")
    check("::ffff:100.90.226.110", "::ffff:100.64.0.9")
    for allow in (False, True):    # Tailscale down here: the packets leave through the default route
        with pytest.raises(WorkerError, match="Tailscale is not connected on this machine"):
            check("100.90.226.110", "192.168.1.20", allow_any_host=allow)
    with pytest.raises(WorkerError, match="not a Tailscale"):
        check("192.168.1.5", "192.168.1.20")
    check("192.168.1.5", "192.168.1.20", allow_any_host=True)


def test_route_checked_on_every_connection_before_the_token(monkeypatch):
    received: list[bytes] = []
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.2)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.settimeout(2.0)
            try:
                received.append(conn.recv(65536))
            except OSError:
                received.append(b"?")
            conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    calls = []

    def refuse(peer, local, allow_any_host=False):
        calls.append((peer, local))
        raise WorkerError("Tailscale is not connected on this machine (test)")

    monkeypatch.setattr(client_mod, "check_route", refuse)
    c = WorkerClient(f"http://127.0.0.1:{srv.getsockname()[1]}", TOKEN)
    for _ in range(2):              # every request, not once per client
        with pytest.raises(WorkerError, match="not connected"):
            c.info()
    wait_for(lambda: len(received) == 2, timeout=10)
    stop.set()
    t.join(5)
    srv.close()
    assert len(calls) == 2 and received == [b"", b""]     # the connection closed before a byte was sent


def test_worker_reported_names_never_become_paths(make_worker, qb_settings, tmp_path):
    w = make_worker("w1")
    w.name = "../.."                                       # what a compromised worker could answer
    (ref,) = client_mod.discover([w.url]).values()
    assert not ref.reachable and "cannot name a folder" in ref.error and "/" not in ref.name
    # a grid.json (a synced file) with such a name: nothing outside the grid folder is touched
    runs = Path(qb_settings.output_dir)
    victim = runs / "2026-10-01_090000_important_w1"
    victim.mkdir(parents=True)
    (victim / "summary.csv").write_text("id\nkeep\n", encoding="utf-8")
    share = client_mod.Share(machine="../..", url=w.url, n=1, job_id="0123456789ab", run_folder=victim.name,
                             state="done")
    grid = _grid(runs / "2026-11-05_120000_grid_x", [share], ["keep"])
    assert client_mod.collect(grid, TOKEN, sync_wait=0, log=lambda m: None) == []
    assert (victim / "summary.csv").read_text(encoding="utf-8") == "id\nkeep\n"
    assert "cannot name a folder" in share.error
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("other_run/summary.csv", "x")
    with pytest.raises(WorkerError, match="not the job's run folder"):
        client_mod.safe_extract(zpath, tmp_path / "out", expected_top="my_run")


def test_cancel_survives_a_restart(tmp_path):
    state, out = tmp_path / "state", tmp_path / "runs"
    run_dir = out / "r1"
    run_dir.mkdir(parents=True)
    state.mkdir()
    rec = {"id": "abcdef012345", "label": "x", "run_dir": str(run_dir), "config": {}, "state": "running",
           "created": "2026-11-05T10:00:00+00:00", "cancel_requested": True}
    (state / worker_mod.STATE_FILE).write_text(json.dumps({"format": 1, "jobs": [rec]}), encoding="utf-8")
    m = worker_mod.JobManager(out, "w1", state_dir=state, awake=False)
    m.start()
    try:
        job = m.get("abcdef012345")
        assert job.state == "cancelled" and job.finished and m.queue_state()["queued"] == []
    finally:
        m.stop(5)
    saved = json.loads((state / worker_mod.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["jobs"][0]["state"] == "cancelled"
    assert json.loads((run_dir / worker_mod.JOB_FILE).read_text(encoding="utf-8"))["job"]["state"] == "cancelled"


def test_lane_spec_per_machine(qb_settings, tmp_path):
    s = settings_mod.load()
    other = "other-pc" if profile.machine_name() != "other-pc" else "another-pc"
    s.set_lanes_for("2", other.upper().replace("-", "_"))   # stored under the normalized name
    assert s.dist.machine_lanes == {other: "2"} and s.lanes_for(other) == "2"
    assert s.lanes_for() == "1" and s.dist.lanes == "1"     # this machine: the shared default
    settings_mod.save(s)
    assert settings_mod.load().lanes_for(other) == "2"
    # a spec that does not fit this machine (no GPU in the tests): from the settings it falls back to one
    # lane with a warning; given explicitly it is an error
    s.set_lanes_for("gpu:1x4")
    cfg = pipeline.BuildConfig(method="mmff", gpu="off")
    logs: list[str] = []
    lanes, spec = profile.resolve_lanes(None, cfg, log=logs.append, settings=s)
    assert spec == "1" and len(lanes) == 1 and "does not fit" in logs[0]
    with pytest.raises(ValueError):
        profile.resolve_lanes("gpu:1x4", cfg, settings=s)
    settings_mod.save(s)
    assert cli.main(["worker", "--bind", "localhost", "--port", "0", "--lanes", "gpu:1x4",
                     "--state-dir", str(tmp_path / "st")]) == 2


def test_worker_job_runs_in_one_lane_when_the_setting_does_not_fit(make_worker, qb_settings):
    s = settings_mod.load()
    s.set_lanes_for("gpu:1x4")              # this machine's entry; the GPU is off in the tests
    settings_mod.save(s)
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    j = c.submit(inputs(["ethanol"]), dict(CONFIG), "lanes")
    st = c.wait(j, poll=0.1, timeout=120)
    assert st["state"] == "done" and st["counts"]["ok"] == 1, st
    assert any("does not fit" in line for line in c.status(j, tail=100)["log_tail"])
    body = client_mod.job_body(inputs(["ethanol"]), dict(CONFIG), "lanes2", lanes="gpu:1x4")
    status, data = raw(w.url + "/v1/jobs", "POST", body)       # a job's own spec must fit
    assert status == 400 and "lanes" in data["error"]


def test_ids_differing_only_in_case():
    from quantabench.chem import jobs
    from quantabench.chem.inputs import dedupe_ids

    benzene, cyclohexane = from_smiles("c1ccccc1"), from_smiles("C1CCCCC1")
    assert benzene.id.casefold() == cyclohexane.id.casefold()   # one folder on NTFS and APFS
    warnings: list[str] = []
    out = dedupe_ids([benzene, cyclohexane, benzene], warnings)
    assert [x.id for x in out] == ["c1ccccc1", "C1CCCCC1_2"] and "letter case" in warnings[0]
    with pytest.raises(worker_mod.RequestError, match="letter case"):
        worker_mod.validate_inputs([jobs.input_to_json(benzene), jobs.input_to_json(cyclohexane)])


def test_surface_options_travel_to_a_worker():
    """The radii set, the per-element radii and the SASA probe of the client's settings reach the worker's
    build, so every machine of a grid computes the same areas and volumes."""
    from dataclasses import asdict

    cfg = asdict(pipeline.BuildConfig(radii_set="spartan", radii_overrides={"C": 1.8, "17": 1.9}, sasa_probe_A=1.0))
    cfg.pop("output_dir")
    out = worker_mod.validate_config(cfg)
    assert (out["radii_set"], out["radii_overrides"], out["sasa_probe_A"]) == ("spartan", {"C": 1.8, "17": 1.9}, 1.0)
    for bad, match in (({"radii_set": "uff"}, "one of"), ({"radii_overrides": {"Xx": 1.5}}, "not an element"),
                       ({"radii_overrides": {"C": 0}}, "radius"), ({"radii_overrides": {"C": True}}, "radius"),
                       ({"radii_overrides": ["C"]}, "object"), ({"sasa_probe_A": -1.0}, "in \\[")):
        with pytest.raises(worker_mod.RequestError, match=match):
            worker_mod.validate_config(bad)


def test_follow_stops_on_a_lost_job_and_retries_an_unreachable_one(make_worker, tmp_path):
    w = make_worker("w1")
    share = client_mod.Share(machine="w1", url=w.url, n=1, job_id="0123456789ab", state="running")
    grid = _grid(tmp_path / "g", [share], ["a"])
    logs: list[str] = []
    client_mod.follow(grid, {"w1": client_mod._RemoteTarget(WorkerClient(w.url, TOKEN), share.job_id)},
                      poll=0.05, log=logs.append)          # returns: the worker does not know the job (404)
    assert share.state == client_mod.LOST and "404" in share.error

    class Down:
        def status(self):
            raise WorkerError("cannot reach it")

    share.state, checks = "running", []
    client_mod.follow(grid, {"w1": Down()}, poll=0.01, log=logs.append,
                      should_stop=lambda: checks.append(1) or len(checks) > 3)
    assert share.state == "running" and len(checks) == 4      # still waited for


def test_partial_merge_and_per_share_errors(make_worker, qb_settings, monkeypatch):
    gate = threading.Event()
    real = pipeline.run_molecule

    def held(inp, cfg, device, threads, should_stop, *a, **k):
        while inp.id == "TFE" and not gate.is_set() and not should_stop():
            time.sleep(0.02)
        return real(inp, cfg, device, threads, should_stop, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", held)
    runs = Path(qb_settings.output_dir)
    w1 = make_worker("w1", output=runs)               # synced: its run folder is in the client's Runs
    c = WorkerClient(w1.url, TOKEN)
    j1 = c.submit(inputs(["ethanol", "TFE"]), dict(CONFIG), "part")
    wait_for(lambda: c.status(j1)["counts"].get("ok") == 1, timeout=120)
    st1 = c.status(j1)
    # a share run by "local" on the other machine: its run_dir is that machine's path, its folder synced here
    local_dir = runs / "2026-11-05_120000_part_mac-local"
    local_dir.mkdir()
    (local_dir / "summary.csv").write_text("id,status\nmethanol,ok\n", encoding="utf-8")
    tables = worker_mod.JobManager._final_tables(local_dir)
    (local_dir / worker_mod.JOB_FILE).write_text(json.dumps(
        {"job": {"id": "aaaaaaaaaaaa", "state": "done"}, "tables": tables}), encoding="utf-8")
    shares = [
        client_mod.Share(machine="w1", url=w1.url, n=2, job_id=j1, run_folder=st1["run_folder"], state="running"),
        client_mod.Share(machine="w2", url=f"http://127.0.0.1:{_closed_port()}", n=1, job_id="bbbbbbbbbbbb",
                         run_folder="2026-11-05_120000_part_w2", state="done"),
        client_mod.Share(machine="local", n=1, job_id="aaaaaaaaaaaa", run_folder=local_dir.name, state="done",
                         run_dir="/Users/someone/elsewhere/" + local_dir.name),
    ]
    grid = _grid(runs / "2026-11-05_120000_grid_part", shares, ["ethanol", "TFE", "acetone", "methanol"])
    logs: list[str] = []
    written = client_mod.merge(grid, TOKEN, sync_wait=0, log=logs.append)
    assert written and sorted(summary_ids(grid.folder)) == ["ethanol", "methanol"]
    assert grid.missing == ["TFE", "acetone"]
    assert [s.source for s in shares] == ["partial", "", "synced"] and "no copy" in shares[1].error
    gate.set()
    assert c.wait(j1, poll=0.1, timeout=120)["state"] == "done"


def test_archive_manifest_sync_temp_files_and_failed_streams(make_worker, tmp_path, monkeypatch):
    w = make_worker("w1")
    c = WorkerClient(w.url, TOKEN)
    j = c.submit(inputs(["ethanol"]), dict(CONFIG), "arch")
    st = c.wait(j, poll=0.1, timeout=120)
    run_dir = Path(st["run_dir"])
    assert not list(run_dir.glob("jobs.sqlite-*"))     # the worker leaves the job table as one file
    (run_dir / "~syncthing~summary.csv.tmp").write_text("partial", encoding="utf-8")   # SyncThing at work
    (run_dir / ".syncthing.x.tmp").write_text("partial", encoding="utf-8")
    dest = c.download_archive(j, tmp_path / "dl", expected_top=st["run_folder"])
    names = {p.name for p in dest.rglob("*")}
    assert worker_mod.ARCHIVE_MANIFEST in names and not any(n.startswith(("~syncthing~", ".syncthing.")) for n in names)
    # an archive without its manifest (a cut-off stream, an older worker) is refused
    zpath = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("run/summary.csv", "id\n")
    with pytest.raises(WorkerError, match="no manifest"):
        client_mod.safe_extract(zpath, tmp_path / "x", require_manifest=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("run/summary.csv", "id\n")
        zf.writestr(f"run/{worker_mod.ARCHIVE_MANIFEST}", json.dumps({"files": [
            {"path": "summary.csv", "size": 3}, {"path": "descriptors.csv", "size": 9}]}))
    with pytest.raises(WorkerError, match="does not match"):
        client_mod.safe_extract(zpath, tmp_path / "x", require_manifest=True)
    # a file that vanishes while the worker zips: no central directory goes out, the client refuses it
    real_members = worker_mod.archive_members
    monkeypatch.setattr(worker_mod, "archive_members", lambda d: [*real_members(d), d / "vanished.xyz"])
    with pytest.raises(WorkerError, match="incomplete or damaged"):
        c.download_archive(j, tmp_path / "dl2")


def test_stale_worker_lock_with_a_reused_pid(tmp_path):
    import os

    import psutil

    state = tmp_path / "state"
    state.mkdir()
    other = os.getppid()                               # a live process that is not this worker
    started = psutil.Process(other).create_time()
    lock = state / worker_mod.STATE_LOCK
    lock.write_text(json.dumps({"pid": other, "pid_started": started - 1000.0, "host": socket.gethostname()}))
    m = worker_mod.JobManager(tmp_path / "runs", "w1", state_dir=state, awake=False)
    m.start()                                          # the pid was reused: the lock is taken over
    m.stop(5)
    lock.write_text(json.dumps({"pid": other, "pid_started": started, "host": socket.gethostname()}))
    with pytest.raises(RuntimeError, match="another worker"):
        worker_mod.JobManager(tmp_path / "runs", "w1", state_dir=state, awake=False).start()


def test_bind_retried_while_the_address_is_not_up(monkeypatch):
    import errno

    monkeypatch.setattr(worker_mod, "BIND_RETRY_S", 0.0)
    tries = []

    def make():
        tries.append(1)
        if len(tries) < 3:
            raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
        return "worker"

    assert worker_mod.bind_with_retry(make, 30.0) == "worker" and len(tries) == 3

    def never():
        raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")

    def busy():
        raise OSError(errno.EADDRINUSE, "Address already in use")

    for make_fail, wait in ((never, 0.0), (busy, 30.0)):    # out of time; not a "not up yet" error
        with pytest.raises(OSError):
            worker_mod.bind_with_retry(make_fail, wait)


def test_commit_mismatch_warns_and_is_recorded(make_worker, qb_settings, monkeypatch):
    w = make_worker("w1", output=Path(qb_settings.output_dir))
    w.commit = "0123456789ab"
    monkeypatch.setattr(profile, "git_commit", lambda root=None: "fedcba987654")
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                               formats=("xyz",), output_dir=qb_settings.output_dir)
    logs: list[str] = []
    grid = client_mod.grid_build(inputs(["ethanol"]), cfg, ["w1"], "1", wait=True, poll=0.2, sync_wait=0,
                                 extra_workers=[f"w1={w.url}"], log=logs.append)
    assert grid.missing == [] and any("runs commit 0123456789ab" in m for m in logs)
    saved = client_mod.GridRun.load(Path(grid.folder) / client_mod.GRID_FILE)
    assert saved.commit == "fedcba987654" and saved.shares[0].commit == "0123456789ab"
    assert client_mod.same_commit("0123456789ab", "0123456789abcdef") and client_mod.same_commit("", "abc")
    assert not client_mod.same_commit("0123456789ab", "fedcba987654")


def test_synced_copy_needs_its_final_tables(tmp_path):
    from quantabench.chem import jobs

    run = tmp_path / "runs" / "2026-11-05_120000_t_w1"
    run.mkdir(parents=True)
    (run / "summary.csv").write_text("id,status\na,ok\n", encoding="utf-8")
    (run / "descriptors.csv").write_text("id,x\na,1\n", encoding="utf-8")
    tables = worker_mod.JobManager._final_tables(run)
    job = {"id": "0123456789ab", "state": "done"}
    (run / worker_mod.JOB_FILE).write_text(json.dumps({"job": job, "tables": tables}), encoding="utf-8")
    assert client_mod.final_tables(run, "0123456789ab") == tables
    assert client_mod.final_tables(run, "ffffffffffff") is None                 # another job's folder
    (run / "descriptors.csv").write_text("id,x\n", encoding="utf-8")             # SyncThing not done yet
    assert client_mod.final_tables(run, "0123456789ab") is None
    (run / "descriptors.csv").write_text("id,x\na,1\n", encoding="utf-8")
    (run / jobs.LOCK_NAME).write_text("{}", encoding="utf-8")
    assert client_mod.final_tables(run, "0123456789ab") is None
    (run / jobs.LOCK_NAME).unlink()
    (run / worker_mod.JOB_FILE).write_text(json.dumps({"job": {**job, "state": "running"}, "tables": tables}),
                                           encoding="utf-8")
    assert client_mod.final_tables(run, "0123456789ab") is None
    # a molecule that crashed the process twice has no summary row: recorded, not "missing"
    crash = tmp_path / "runs" / "2026-11-05_120000_c_w1"
    with jobs.JobStore(crash) as store:
        store.init_run(pipeline.BuildConfig(method="mmff"), None)
        store.add_inputs(inputs(["ethanol"]))
        for _ in range(jobs.MAX_CRASHES):
            store.mark_running("ethanol")
            store.reset_interrupted()
    tables = worker_mod.JobManager._final_tables(crash, rebuild=True)
    assert [x[0] for x in tables["unrecorded_failures"]] == ["ethanol"] and tables["files"] == {}
    (crash / worker_mod.JOB_FILE).write_text(json.dumps({"job": job, "tables": tables}), encoding="utf-8")
    share = client_mod.Share(machine="w1", url="", n=1, job_id="0123456789ab", run_folder=crash.name, state="done")
    grid = _grid(tmp_path / "runs" / "2026-11-05_120000_grid_c", [share], ["ethanol"])
    client_mod.merge(grid, TOKEN, sync_wait=0, log=lambda m: None)
    assert grid.failed_without_result == ["ethanol"] and grid.missing == []


def test_reading_another_machines_copy_leaves_it_unchanged(tmp_path):
    from quantabench.chem import jobs

    run = tmp_path / "run"
    with jobs.JobStore(run) as store:
        store.init_run(pipeline.BuildConfig(method="mmff"), None)
        store.add_inputs(inputs(["ethanol"]))
    before = sorted(p.name for p in run.iterdir())
    assert client_mod.copy_counts(run)["pending"] == 1
    assert client_mod._snapshot(run, tmp_path / "parts") is None          # nothing finished yet
    assert sorted(p.name for p in run.iterdir()) == before == ["jobs.sqlite"]


def test_ensure_token_never_overwrites_unreadable_settings(qb_settings):
    path = settings_mod.settings_dir() / settings_mod.SETTINGS_FILE
    half = '{"output_dir": "~/SyncThing/QuantaBench/Runs", "dist": {'
    path.write_text(half, encoding="utf-8")                                 # half an edit, or a bad merge
    with pytest.raises(ValueError, match="cannot read"):
        worker_mod.ensure_token()
    assert cli.main(["worker", "--bind", "localhost", "--port", "0"]) == 2
    assert path.read_text(encoding="utf-8") == half


def test_job_list_survives_a_torn_file_and_save_errors(tmp_path, monkeypatch):
    state, out = tmp_path / "state", tmp_path / "runs"
    m = worker_mod.JobManager(out, "w1", state_dir=state, awake=False)
    m.paused = True
    m.start()
    job = m.submit(worker_mod.validate_job(client_mod.job_body(inputs(["ethanol"]), dict(CONFIG), "t"), __version__))
    m.stop(5)
    assert (state / worker_mod.STATE_BACKUP).is_file()
    (state / worker_mod.STATE_FILE).write_bytes(b"\x00" * 64)               # torn by a power cut
    m2 = worker_mod.JobManager(out, "w1", state_dir=state, awake=False)
    m2.paused = True
    m2.start()
    try:
        assert m2.get(job.id) is not None and m2.queue_state()["queued"] == [job.id]
        assert (state / (worker_mod.STATE_FILE + ".unreadable")).is_file()
    finally:
        m2.stop(5)
    # saving the job list fails (a file held open on Windows): the job still runs, the thread survives
    real = worker_mod._atomic_write

    def flaky(path, text, durable=False):
        if Path(path).name.startswith(worker_mod.STATE_FILE):
            raise PermissionError("held by another process")
        return real(path, text, durable)

    monkeypatch.setattr(worker_mod, "_atomic_write", flaky)
    m3 = worker_mod.JobManager(out, "w1", state_dir=state, awake=False)
    m3.start()
    try:
        wait_for(lambda: m3.get(job.id).state == "done", timeout=120)
        assert all(t.is_alive() for t in m3._threads)
    finally:
        m3.stop(5)


# ------------------------------------------------------------------------------- join-time benchmark
def test_new_machine_benchmarks_once_with_jobs_held(make_worker, qb_settings, monkeypatch):
    calls, gate = [], threading.Event()
    real = profile.local_profile

    def fake_local_profile(bench=False, methods=None, **kw):
        prof = real(**{k: v for k, v in kw.items() if k in ("address", "keep_previous")})
        if bench:
            calls.append(methods)
            gate.wait(10)                      # the test checks the hold while the "benchmark" runs
            prof.throughput = {"mmff@cpu": 1234.0}
        return prof

    monkeypatch.setattr(profile, "local_profile", fake_local_profile)
    w = make_worker("joiner", bench="auto", publish=True)
    c = WorkerClient(w.url, TOKEN)
    assert c.info()["queue"]["hold"] == "benchmarking this machine"
    job = c.submit(inputs(["ethanol"]), dict(CONFIG), "held")
    time.sleep(0.5)
    assert c.status(job)["state"] == "queued"                   # waits for the benchmark
    gate.set()
    assert c.wait(job, poll=0.1, timeout=60)["state"] == "done"
    assert calls == [None] and c.info()["queue"]["hold"] == ""
    assert w.profile.throughput == {"mmff@cpu": 1234.0}
    saved = {p.name: p for p in profile.load_profiles()}["joiner"]
    assert saved.throughput == {"mmff@cpu": 1234.0} and saved.address == w.url
    w.stop(timeout=30)
    # a machine that already has a measurement does not benchmark again on its next start
    again = make_worker("joiner", bench="auto", state=Path(qb_settings.output_dir).parent / "joiner2_state")
    assert not again.needs_benchmark() and again.info()["queue"]["hold"] == "" and calls == [None]


def test_benchmark_modes():
    assert worker_mod.BENCH_MODES == ("auto", "always", "never")
    with pytest.raises(ValueError, match="bench must be"):
        worker_mod.Worker("127.0.0.1", 0, TOKEN, "out", bench="sometimes")
