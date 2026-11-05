"""Workers and grid builds (quantabench.dist.worker / dist.client), all on 127.0.0.1 with threads.

Each worker gets its own port (0 = any free one), state folder and output folder; the settings folder is the
per-test one from conftest.py. Molecules are tiny and run with MMFF, without the xTB property step.
"""

import csv
import json
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
    assert info["queue"] == {"running": [], "queued": [], "paused": False, "max_jobs": 1}
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
