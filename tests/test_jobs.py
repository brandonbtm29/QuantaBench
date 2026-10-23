import csv
import json
import sqlite3

import pytest

from quantabench.chem import jobs, pipeline
from quantabench.chem.inputs import from_smiles

MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F")]


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1, gpu="off",
                output_dir=str(tmp_path / "runs"), formats=("xyz",))
    base.update(kw)
    return pipeline.BuildConfig(**base)


def inputs(names=None):
    return [from_smiles(s, id=i) for i, s in MOLS if names is None or i in names]


@pytest.fixture
def counting(monkeypatch):
    """Wrap pipeline.run_molecule and record which ids it was called for."""
    calls = []
    real = pipeline.run_molecule

    def wrapped(inp, *a, **k):
        calls.append(inp.id)
        return real(inp, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", wrapped)
    return calls


def summary_ids(run_dir):
    with open(run_dir / "summary.csv", newline="", encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f)]


def test_interrupt_then_resume(tmp_path, counting):
    run_dir = tmp_path / "run"
    stop = {"now": False}

    def on_result(res):
        stop["now"] = True  # ask to stop as soon as the first molecule is done

    results, _ = jobs.run_resumable(inputs(), cfg(tmp_path), run_dir, should_stop=lambda: stop["now"],
                                    on_result=on_result)
    assert [r.input.id for r in results] == ["ethanol"] and counting == ["ethanol"]
    st = jobs.open_run(run_dir)
    assert st.counts["ok"] == 1 and st.counts["pending"] == 3 and not st.active
    assert summary_ids(run_dir) == ["ethanol"]

    counting.clear()
    seen = []
    results, _ = jobs.run_resumable(inputs(), cfg(tmp_path), run_dir,
                                    progress=lambda i, n, stage, ident: seen.append((i, n, stage)))
    assert counting == ["acetic_acid", "TFE", "fluoroform"]  # ethanol is not redone
    assert all(r.status == "ok" for r in results), [r.error for r in results]
    assert seen[0] == (1, 4, "start") and seen[-1][:2] == (4, 4)
    assert summary_ids(run_dir) == [i for i, _ in MOLS]
    with open(run_dir / "descriptors.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 4
    assert (run_dir / "acetic_acid" / "acetic_acid.xyz").exists() and not (run_dir / jobs.LOCK_NAME).exists()
    assert jobs.open_run(run_dir).summary().startswith("4/4 done")

    # nothing left: a third call does no work; resuming from the table alone (inputs=None) works too
    counting.clear()
    jobs.run_resumable(None, cfg(tmp_path), run_dir)
    assert counting == []


def test_retry_failed(tmp_path, monkeypatch):
    real = pipeline.run_molecule
    calls = []
    failures_left = {"acetic_acid": 1}

    def flaky(inp, *a, **k):
        calls.append(inp.id)
        if failures_left.get(inp.id):
            failures_left[inp.id] -= 1
            return pipeline.MoleculeResult(input=inp, status="failed", error="RuntimeError: simulated")
        return real(inp, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", flaky)
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol", "acetic_acid"]), cfg(tmp_path), run_dir)
    st = jobs.open_run(run_dir)
    assert st.counts["ok"] == 1 and st.counts["failed"] == 1
    assert st.failures == [("acetic_acid", "RuntimeError: simulated")]
    # failed rows appear in summary.csv with their status, so a batch table shows what went wrong
    assert summary_ids(run_dir) == ["ethanol", "acetic_acid"]

    calls.clear()
    jobs.run_resumable(inputs(["ethanol", "acetic_acid"]), cfg(tmp_path), run_dir)
    assert calls == []  # failures are not retried by default
    jobs.run_resumable(inputs(["ethanol", "acetic_acid"]), cfg(tmp_path), run_dir, retry_failed=True)
    assert calls == ["acetic_acid"]
    assert jobs.open_run(run_dir).counts["ok"] == 2
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert con.execute("SELECT attempts FROM molecules WHERE id='acetic_acid'").fetchone()[0] == 2


def test_chunks_partition():
    ids = [f"DTXSID{n:07d}" for n in range(60)]
    for n in (1, 2, 3, 5):
        parts = [{i for i in ids if jobs.in_chunk(i, (k, n))} for k in range(n)]
        assert sum(len(p) for p in parts) == len(ids)
        assert set().union(*parts) == set(ids)
    assert jobs.parse_chunk("2/3") == (1, 3) and jobs.chunk_label((1, 3)) == "chunk2of3"
    with pytest.raises(ValueError):
        jobs.check_chunk((3, 3))
    with pytest.raises(ValueError):
        jobs.parse_chunk("0/3")


def test_chunked_runs_cover_everything(tmp_path, counting):
    dirs = []
    for k in range(2):
        d = tmp_path / f"run_{k}"
        jobs.run_resumable(inputs(), cfg(tmp_path), d, chunk=(k, 2))
        dirs.append(d)
    assert sorted(counting) == sorted(i for i, _ in MOLS)  # each molecule exactly once
    per_dir = [set(summary_ids(d)) if (d / "summary.csv").exists() else set() for d in dirs]
    assert not per_dir[0] & per_dir[1] and per_dir[0] and per_dir[1]  # these ids hash to both chunks
    # a resume without chunk keeps the stored chunk; a different chunk is refused
    counting.clear()
    jobs.run_resumable(inputs(), cfg(tmp_path), dirs[0])
    assert counting == []
    with pytest.raises(jobs.ConfigMismatchError):
        jobs.run_resumable(inputs(), cfg(tmp_path), dirs[0], chunk=(1, 2))
    merged = jobs.merge_tables(dirs, tmp_path / "merged")
    assert sorted(summary_ids(tmp_path / "merged")) == sorted(i for i, _ in MOLS)
    assert merged[0].name == "summary.csv"


def test_config_mismatch_guard(tmp_path, counting):
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), run_dir)
    with pytest.raises(jobs.ConfigMismatchError) as err:
        jobs.run_resumable(inputs(), cfg(tmp_path, ph=5.0, energy_window_kcal=3.0), run_dir)
    assert set(err.value.differences) == {"ph", "energy_window_kcal"}
    assert "force=True" in str(err.value)
    assert not (run_dir / jobs.LOCK_NAME).exists()  # the refusal released the lock
    # speed/output settings may change freely
    jobs.run_resumable(inputs(["ethanol", "acetic_acid"]), cfg(tmp_path, threads=2, formats=("xyz", "sdf")), run_dir)
    assert counting == ["ethanol", "acetic_acid"]
    jobs.run_resumable(inputs(), cfg(tmp_path, ph=5.0), run_dir, force=True)
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert json.loads(con.execute("SELECT config FROM run").fetchone()[0])["ph"] == 5.0


def test_crash_leaves_running_row_which_is_resumed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    real = pipeline.run_molecule

    def crash_on_second(inp, *a, **k):
        if inp.id == "acetic_acid":
            raise KeyboardInterrupt  # stands in for a kill/reboot in the middle of a molecule
        return real(inp, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        jobs.run_resumable(inputs(["ethanol", "acetic_acid", "TFE"]), cfg(tmp_path), run_dir)
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert con.execute("SELECT status FROM molecules WHERE id='acetic_acid'").fetchone()[0] == "running"
    # a hard crash also leaves the lock file behind, owned by a process that no longer exists
    (run_dir / jobs.LOCK_NAME).write_text(json.dumps({"host": jobs.socket.gethostname(), "pid": 2**22 + 7,
                                                       "pid_started": 1.0, "since": "earlier"}))
    st = jobs.open_run(run_dir)
    assert st.interrupted == 1 and st.counts["running"] == 0 and st.counts["pending"] == 2 and not st.active

    monkeypatch.setattr(pipeline, "run_molecule", real)
    logs = []
    jobs.run_resumable(None, cfg(tmp_path), run_dir, log=logs.append)
    assert any("interrupted by a crash" in m for m in logs)
    assert jobs.open_run(run_dir).counts["ok"] == 3
    assert summary_ids(run_dir) == ["ethanol", "acetic_acid", "TFE"]


def test_live_lock_refused(tmp_path):
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), run_dir)
    (run_dir / jobs.LOCK_NAME).write_text(json.dumps({"host": "other-pc", "pid": 1, "since": "now"}))
    with pytest.raises(jobs.RunInUseError):
        jobs.run_resumable(None, cfg(tmp_path), run_dir)
    jobs.run_resumable(None, cfg(tmp_path), run_dir, force=True)


def test_stored_input_round_trip_keeps_structure_files():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("[O-]C(=O)C(F)(F)F"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    inp = from_smiles("CCO", id="x")
    inp.mol, inp.meta = m, {"DTXSID": "x", "row": 2}
    back = jobs.input_from_json(jobs.input_to_json(inp))
    assert back.mol.GetNumAtoms() == m.GetNumAtoms() and back.meta == inp.meta
    assert Chem.GetFormalCharge(back.mol) == -1


def _snapshot(folder):
    import hashlib

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())
            if p.is_file() and p.name.startswith("jobs.sqlite") and not p.name.endswith("-shm")}


def test_status_does_not_modify_a_synced_copy(tmp_path):
    """open_run must not checkpoint the WAL or delete it: SyncThing would send that back to the machine
    still writing the database."""
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol", "TFE"]), cfg(tmp_path), run_dir)
    # stand-in for a copy taken mid-run: a writer connection with uncheckpointed commits in the -wal
    writer = sqlite3.connect(run_dir / jobs.DB_NAME)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    with writer:
        writer.execute("UPDATE molecules SET status = 'pending' WHERE id = 'TFE'")
    assert (run_dir / (jobs.DB_NAME + "-wal")).exists()
    before = _snapshot(run_dir)
    st = jobs.open_run(run_dir)
    assert st.counts["ok"] == 1 and st.counts["pending"] == 1      # reads the -wal
    assert _snapshot(run_dir) == before
    writer.close()


def test_molecule_that_crashes_twice_is_marked_failed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    real = pipeline.run_molecule
    calls = []

    def crash(inp, *a, **k):
        calls.append(inp.id)
        if inp.id == "acetic_acid":
            raise KeyboardInterrupt      # stands in for a segfault / OOM kill
        return real(inp, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", crash)
    with pytest.raises(KeyboardInterrupt):
        jobs.run_resumable(inputs(["ethanol", "acetic_acid", "TFE"]), cfg(tmp_path), run_dir)
    assert calls == ["ethanol", "acetic_acid"]
    # 1st crash: the crashed molecule goes last, so the rest of the list still gets done
    with pytest.raises(KeyboardInterrupt):
        jobs.run_resumable(None, cfg(tmp_path), run_dir)
    assert calls[2:] == ["TFE", "acetic_acid"]
    # 2nd crash: marked failed, the run completes
    logs = []
    jobs.run_resumable(None, cfg(tmp_path), run_dir, log=logs.append)
    st = jobs.open_run(run_dir)
    assert st.counts["ok"] == 2 and st.counts["failed"] == 1 and st.remaining == 0
    assert st.failures == [("acetic_acid", "crashed the process 2 times")]
    assert any("crashed the process" in m for m in logs)
    # --retry-failed still redoes it
    monkeypatch.setattr(pipeline, "run_molecule", real)
    jobs.run_resumable(None, cfg(tmp_path), run_dir, retry_failed=True)
    assert jobs.open_run(run_dir).counts["ok"] == 3


def test_lock_from_another_os_on_the_same_host_needs_force(tmp_path, monkeypatch):
    """WSL2 shares the Windows host name but not its pids: a lock from the other OS is 'unknown'."""
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), run_dir)
    import os

    other = "Windows" if jobs._platform_id() != "Windows" else "Linux-wsl"
    (run_dir / jobs.LOCK_NAME).write_text(json.dumps({"host": jobs.socket.gethostname(), "platform": other,
                                                       "pid": os.getpid() + 12345, "since": "now"}))
    assert jobs._lock_alive(jobs._read_lock(run_dir)) is None
    with pytest.raises(jobs.RunInUseError):
        jobs.run_resumable(None, cfg(tmp_path), run_dir)
    jobs.run_resumable(None, cfg(tmp_path), run_dir, force=True)
    assert not (run_dir / jobs.LOCK_NAME).exists()


def test_merge_reports_missing_and_unfinished_folders(tmp_path):
    done, killed = tmp_path / "done", tmp_path / "killed"
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), done)
    jobs.run_resumable(inputs(["TFE"]), cfg(tmp_path), killed)
    (killed / "summary.csv").unlink()            # a chunk killed before it rewrote its tables
    (killed / "descriptors.csv").unlink()
    logs = []
    written = jobs.merge_tables([done, killed, tmp_path / "nope"], tmp_path / "merged", log=logs.append)
    assert {p.name for p in written} == {"summary.csv", "descriptors.csv"}
    assert summary_ids(tmp_path / "merged") == ["ethanol", "TFE"]
    assert any("no summary.csv" in m and "killed" in m for m in logs)
    assert any("no such folder" in m for m in logs)
