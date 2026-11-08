"""Run telemetry (chem.telemetry) and the timings command (dist.timings): the job table's new tables and their
migration, what a sequential run, a lanes run, a crashed or killed lane and a resumed run record, the CPU time of a
program, the aggregator over run folders and grid parts, and the CREST cost-model fit."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sqlite3
import sys
import textwrap
import threading

import pytest

from quantabench import settings as settings_mod
from quantabench.chem import external, jobs, pipeline, telemetry
from quantabench.chem.inputs import from_smiles
from quantabench.dist import lanes as L
from quantabench.dist import threads as T
from quantabench.dist import timings
from quantabench.dist.lanes import Lane

MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F")]
PFHPA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"         # 23 atoms with H
PFBA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"                                  # 14
ETHANOL = "CCO"                                                        # 9
POSIX = os.name != "nt"


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1, gpu="off",
                output_dir=str(tmp_path / "runs"), formats=("xyz",))
    base.update(kw)
    return pipeline.BuildConfig(**base)


def inputs(names=None):
    return [from_smiles(s, id=i) for i, s in MOLS if names is None or i in names]


def rows(run_dir, table, where="", args=()):
    con = sqlite3.connect(run_dir / jobs.DB_NAME)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table} {where}", args)]  # noqa: S608
    finally:
        con.close()


def stages_of(run_dir, mol_id, attempt=None):
    where = "WHERE mol_id = ?" + (" AND attempt = ?" if attempt else "") + " ORDER BY attempt, seq"
    return rows(run_dir, "telemetry_stages", where, (mol_id, attempt) if attempt else (mol_id,))


# ------------------------------------------------------------------------------------- fake CREST
FAKE_CREST = r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if "--version" in args:
    print("       Version 3.0.2, fake")
    sys.exit(0)
t0 = time.time()
lines = Path(args[0]).read_text().splitlines()
n = int(lines[0])
if str(n) in os.environ.get("FAKE_CREST_KILL_LANE", "").split(","):
    os.kill(os.getppid(), signal.SIGKILL)            # the lane that started this CREST dies mid-search
    time.sleep(10)
c0 = time.process_time()
while time.process_time() - c0 < float(os.environ.get("FAKE_CREST_BURN", "0")):
    pass
time.sleep(float(os.environ.get("FAKE_CREST_SLEEP", "0")))
Path("crest_conformers.xyz").write_text("\n".join([f"  {n}", "  -11.30"] + lines[2:2 + n]) + "\n")
with open(os.environ["FAKE_CREST_LOG"], "a") as f:
    f.write(json.dumps({"T": int(args[args.index("-T") + 1]), "atoms": n, "t0": t0, "t1": time.time()}) + "\n")
print(" CREST terminated normally.")
'''


@pytest.fixture
def fake_crest(tmp_path, monkeypatch):
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    settings_mod.save(s)
    external.clear_cache()
    folder = tmp_path / "crestbin"
    folder.mkdir()
    script = folder / "crest_impl.py"
    script.write_text(textwrap.dedent(FAKE_CREST), encoding="utf-8")
    if sys.platform == "win32":
        exe = folder / "crest.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = folder / "crest"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    log = tmp_path / "crest_calls.jsonl"
    monkeypatch.setenv("QUANTABENCH_CREST", str(exe))
    monkeypatch.setenv("FAKE_CREST_LOG", str(log))
    yield lambda: [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    external.clear_cache()


def crest_cfg(tmp_path, **kw):
    return cfg(tmp_path, **{"conformer_search": "crest", "crest_method": "gfnff", "threads": 3, "protonate": False,
                            **kw})


def three_lanes():
    return [Lane(f"cpu{i}", "cpu", 1) for i in (1, 2, 3)]


# -------------------------------------------------------------------------------------- the schema
V3_SCHEMA = """
CREATE TABLE run (id INTEGER PRIMARY KEY CHECK (id = 1), config TEXT NOT NULL, version TEXT NOT NULL,
                  created TEXT NOT NULL, chunk TEXT, updated TEXT, accepted TEXT);
CREATE TABLE molecules (id TEXT PRIMARY KEY, seq INTEGER NOT NULL, input TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending', error TEXT NOT NULL DEFAULT '',
                        attempts INTEGER NOT NULL DEFAULT 0, started TEXT, finished TEXT,
                        result_dir TEXT NOT NULL DEFAULT '', summary TEXT, crashes INTEGER NOT NULL DEFAULT 0);
CREATE INDEX molecules_status ON molecules(status);
PRAGMA user_version=3;
"""


def old_run_folder(tmp_path, c, mols):
    """A run folder as QuantaBench 0.3.0.dev3 (schema 3) left it: ethanol done, the others pending."""
    from dataclasses import asdict

    run_dir = tmp_path / "old"
    run_dir.mkdir()
    con = sqlite3.connect(run_dir / jobs.DB_NAME)
    con.executescript(V3_SCHEMA)
    con.execute("INSERT INTO run (id, config, version, created, chunk, updated) VALUES (1, ?, ?, ?, NULL, ?)",
                (json.dumps(asdict(c), default=str), "0.3.0.dev3", "2026-11-01T00:00:00+00:00",
                 "2026-11-01T00:00:00+00:00"))
    for seq, m in enumerate(mols):
        done = m.id == "ethanol"
        con.execute("INSERT INTO molecules (id, seq, input, status, attempts, summary) VALUES (?, ?, ?, ?, ?, ?)",
                    (m.id, seq, jobs.input_to_json(m), "ok" if done else "pending", 1 if done else 0,
                     json.dumps({"row": {"id": m.id, "status": "ok"}, "descriptors": {}}) if done else None))
    con.commit()
    con.close()
    return run_dir


def test_an_old_job_table_gets_the_new_tables_and_resumes(tmp_path):
    c = cfg(tmp_path)
    run_dir = old_run_folder(tmp_path, c, inputs())
    with jobs.JobStore(run_dir) as store:
        assert telemetry.has_tables(store.conn)
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == jobs.SCHEMA_VERSION == 4
        assert store.get_run()["version"] == "0.3.0.dev3" and store.counts()["ok"] == 1
    # read-only opens of an old table (status, merge) never create anything; a resume records its timings
    assert jobs.open_run(run_dir).counts["pending"] == 3
    results, _ = jobs.run_resumable(None, c, run_dir, force=True)
    assert sorted(r.input.id for r in results) == ["TFE", "acetic_acid", "fluoroform"]
    run = rows(run_dir, "telemetry_runs")
    assert len(run) == 1 and run[0]["n_done_before"] == 1 and run[0]["n_todo"] == 3
    assert {a["mol_id"] for a in rows(run_dir, "telemetry_attempts")} == {"TFE", "acetic_acid", "fluoroform"}


def test_a_telemetry_table_without_a_newer_column_gets_it(tmp_path):
    con = sqlite3.connect(tmp_path / "t.sqlite")
    con.execute("CREATE TABLE telemetry_stages (run_id TEXT NOT NULL, mol_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
                "stage TEXT NOT NULL, wall_s REAL, PRIMARY KEY (run_id, mol_id, attempt, stage))")
    telemetry.create_tables(con)
    cols = {r[1] for r in con.execute("PRAGMA table_info(telemetry_stages)")}
    assert {c for c, _ in telemetry.STAGE_COLUMNS} <= cols
    telemetry.create_tables(con)                       # twice is fine
    assert telemetry.has_tables(con)


# -------------------------------------------------------------------------------- sequential runs
def test_a_sequential_run_records_its_machine_attempts_and_stages(tmp_path):
    logs = []
    results, run_dir = jobs.run_resumable(inputs(), cfg(tmp_path), tmp_path / "run", log=logs.append)
    assert all(r.status == "ok" for r in results)
    [run] = rows(run_dir, "telemetry_runs")
    assert run["machine"] and run["cpu_model"] and run["logical_cores"] >= 1 and run["ram_gb"] > 0
    assert run["qb_version"] and run["os"] and run["method"] == "mmff" and run["n_todo"] == 4
    assert run["thread_budget"] == 1 and run["lanes_text"].startswith("1 lane") and run["dynamic_threads"] == 0
    assert run["ended_unix"] >= run["started_unix"] and run["n_ok"] == 4 and run["n_attempts"] == 4
    assert "rdkit" in json.loads(run["engines_json"])["packages"]
    assert json.loads(run["config_json"])["max_conformers"] == 2
    attempts = {a["mol_id"]: a for a in rows(run_dir, "telemetry_attempts")}
    assert set(attempts) == {i for i, _ in MOLS}
    e = attempts["ethanol"]
    assert e["attempt"] == 1 and e["outcome"] == "ok" and e["lane"] == "main" and e["lane_threads"] == 1
    assert (e["n_atoms"], e["n_heavy"], e["n_O"], e["n_F"], e["n_rotatable"], e["size"]) == (9, 3, 1, 0, 1, 9.0)
    assert attempts["TFE"]["n_F"] == 3 and attempts["acetic_acid"]["formal_charge"] == -1     # deprotonated at pH 7
    assert e["queue_wait_s"] >= 0 and e["wall_s"] > 0 and e["cpu_s"] >= 0 and e["n_conformers"] >= 1
    st = stages_of(run_dir, "ethanol")
    assert [s["stage"] for s in st] == ["prepare", "conformers", "refine", "descriptors", "export"]
    assert all(s["outcome"] == "ok" and s["wall_s"] >= 0 and s["cpu_s"] is not None and s["threads_given"] == 1
               for s in st)
    assert st[1]["engine"] == "rdkit etkdg + MMFF94s" and st[2]["engine"] == "mmff" and st[2]["n_out"] >= 1
    assert sum(s["wall_s"] for s in st) <= e["wall_s"] + 1e-6
    assert any("Timings of 4 molecule(s) recorded in jobs.sqlite; quantabench-cli timings" in m for m in logs)
    # the result's own files and table stay as they were: no telemetry in the summary JSON
    summary = json.loads((run_dir / "ethanol" / "ethanol_summary.json").read_text(encoding="utf-8"))
    assert "telemetry" not in summary and results[0].telemetry["stages"]


def test_a_failed_molecule_records_the_stage_it_failed_in(tmp_path, monkeypatch):
    from quantabench.chem import descriptors

    real = descriptors.graph_descriptors

    def boom(mol):
        if mol.GetNumAtoms() == 9 and any(a.GetSymbol() == "F" for a in mol.GetAtoms()):
            raise RuntimeError("no descriptors for TFE")
        return real(mol)

    monkeypatch.setattr(descriptors, "graph_descriptors", boom)
    _, run_dir = jobs.run_resumable(inputs(["TFE", "ethanol"]), cfg(tmp_path), tmp_path / "run")
    [a] = rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'TFE'")
    assert a["outcome"] == "failed" and "no descriptors for TFE" in a["error"] and a["last_stage"] == "descriptors"
    st = stages_of(run_dir, "TFE")
    assert [s["stage"] for s in st] == ["prepare", "conformers", "refine", "descriptors"]
    assert st[-1]["outcome"] == "failed" and st[0]["outcome"] == "ok"


def test_telemetry_never_fails_a_run(tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error (test)")

    monkeypatch.setattr(telemetry, "_insert", broken)
    logs = []
    results, run_dir = jobs.run_resumable(inputs(), cfg(tmp_path), tmp_path / "run", log=logs.append)
    assert all(r.status == "ok" for r in results) and jobs.open_run(run_dir).counts["ok"] == 4
    assert sum("Timings: not recorded for this run" in m for m in logs) == 1
    assert rows(run_dir, "telemetry_runs") == [] and rows(run_dir, "telemetry_attempts") == []
    assert not any("quantabench-cli timings" in m for m in logs)


def test_telemetry_write_errors_mid_run_are_logged_once(tmp_path, monkeypatch):
    real = telemetry._insert

    def flaky(conn, table, row):
        if table == "telemetry_stages":
            raise sqlite3.OperationalError("database is full (test)")
        real(conn, table, row)

    monkeypatch.setattr(telemetry, "_insert", flaky)
    logs = []
    results, run_dir = jobs.run_resumable(inputs(), cfg(tmp_path), tmp_path / "run", log=logs.append)
    assert all(r.status == "ok" for r in results) and jobs.open_run(run_dir).counts["ok"] == 4
    assert sum("Timings: could not record" in m for m in logs) == 1
    assert len(rows(run_dir, "telemetry_runs")) == 1 and rows(run_dir, "telemetry_stages") == []


def test_the_switch_turns_recording_off(tmp_path, monkeypatch):
    monkeypatch.setenv(telemetry.ENV_SWITCH, "0")
    results, run_dir = jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), tmp_path / "run")
    assert results[0].status == "ok" and results[0].telemetry == {}
    assert rows(run_dir, "telemetry_runs") == [] and rows(run_dir, "telemetry_stages") == []


# ----------------------------------------------------------------------------------- resumed runs
def test_resumed_runs_count_attempts(tmp_path, monkeypatch):
    """A failed molecule redone with --retry-failed is attempt 2 in a second run row; a molecule interrupted by
    Ctrl+C is closed as 'interrupted' and its next attempt is 2."""
    real = pipeline.run_molecule
    fail = {"acetic_acid"}

    def maybe_fail(inp, *a, **k):
        res = real(inp, *a, **k)
        if inp.id in fail:
            res.status, res.error = "failed", "RuntimeError: once"
        return res

    monkeypatch.setattr(pipeline, "run_molecule", maybe_fail)
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol", "acetic_acid"]), cfg(tmp_path), run_dir)
    fail.clear()
    jobs.run_resumable(None, cfg(tmp_path), run_dir, retry_failed=True)
    runs = sorted(rows(run_dir, "telemetry_runs"), key=lambda r: r["started_unix"])
    assert len(runs) == 2 and runs[1]["n_done_before"] == 1 and runs[1]["n_todo"] == 1
    tries = sorted((a["attempt"], a["outcome"], a["run_id"]) for a in rows(run_dir, "telemetry_attempts")
                   if a["mol_id"] == "acetic_acid")
    assert [(n, o) for n, o, _ in tries] == [(1, "failed"), (2, "ok")]
    assert tries[0][2] == runs[0]["run_id"] and tries[1][2] == runs[1]["run_id"]
    assert {s["attempt"] for s in stages_of(run_dir, "acetic_acid")} == {1, 2}

    def interrupt(inp, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_molecule", interrupt)
    with pytest.raises(KeyboardInterrupt):
        jobs.run_resumable(inputs(["TFE"]), cfg(tmp_path), run_dir)
    [a] = rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'TFE'")
    assert a["attempt"] == 1 and a["outcome"] == "interrupted"
    monkeypatch.setattr(pipeline, "run_molecule", real)
    jobs.run_resumable(None, cfg(tmp_path), run_dir)
    assert sorted((a["attempt"], a["outcome"]) for a in rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'TFE'")) \
        == [(1, "interrupted"), (2, "ok")]
    assert len(rows(run_dir, "telemetry_runs")) == 4


def test_a_running_row_left_by_a_dead_process_becomes_interrupted(tmp_path):
    run_dir = tmp_path / "run"
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), run_dir)
    con = sqlite3.connect(run_dir / jobs.DB_NAME)       # what a power cut during attempt 2 leaves
    con.execute("INSERT INTO telemetry_attempts (run_id, mol_id, attempt, outcome) VALUES ('dead', 'TFE', 1, 'running')")
    con.commit()
    con.close()
    jobs.run_resumable(inputs(["TFE"]), cfg(tmp_path), run_dir)
    assert rows(run_dir, "telemetry_attempts", "WHERE run_id = 'dead'")[0]["outcome"] == "interrupted"


# ----------------------------------------------------------------------------------- programs
BURN = "import time\nt = time.process_time()\nwhile time.process_time() - t < {s}:\n    pass\n"


def test_the_cpu_time_of_a_program_is_captured(tmp_path):
    n0 = external.program_count()
    run = external.run_program([sys.executable, "-c", BURN.format(s=0.4)], tmp_path, threads=3)
    assert run.cpu_s is not None and 0.35 <= run.cpu_s <= run.seconds + 0.2, (run.cpu_s, run.seconds)
    [rec] = external.programs_since(n0)
    assert rec["threads"] == 3 and rec["cpu_s"] == run.cpu_s and rec["name"].lower().startswith("python")
    # a stage around it: its own CPU time and the program's are kept apart
    res = pipeline.MoleculeResult(input=from_smiles("CCO", id="x"))
    clock = telemetry.StageClock(res, 2)
    clock.start("qm", engine="test")
    external.run_program([sys.executable, "-c", BURN.format(s=0.3)], tmp_path, threads=2)
    clock.stop(n_out=1)
    [s] = res.telemetry["stages"]
    assert s["programs"] == 1 and s["program_threads"] == 2 and s["child_cpu_s"] >= 0.25
    assert s["cpu_s"] < 0.2 and s["wall_s"] >= 0.3


@pytest.mark.skipif(not POSIX, reason="the overlap rule is for getrusage(RUSAGE_CHILDREN), POSIX only")
def test_programs_that_overlap_in_one_process_have_no_cpu_time(tmp_path):
    out = {}

    def one(name, s):
        out[name] = external.run_program([sys.executable, "-c", BURN.format(s=s)], tmp_path / name).cpu_s

    threads = [threading.Thread(target=one, args=(f"p{i}", 0.5)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == {"p0": None, "p1": None}
    assert external.run_program([sys.executable, "-c", "pass"], tmp_path / "p3").cpu_s is not None


def test_program_names():
    assert external.program_name(["/opt/x/bin/crest", "input.xyz"]) == "crest"
    assert external.program_name([r"C:\xtb\bin\xtb.exe", "-T", "2"]) == "xtb"
    assert external.program_name(["wsl.exe", "--cd", "/mnt/c/x", "-e", "sh", "-c", external.WSL_WRAPPER,
                                  "/home/b/miniforge3/envs/qb-tools/bin/crest", "input.xyz"]) == "crest"


def test_a_crest_search_in_one_process_records_its_threads_and_cpu(tmp_path, fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_BURN", "0.3")
    results, run_dir = jobs.run_resumable([from_smiles(PFBA, id="PFBA")], crest_cfg(tmp_path, threads=2),
                                          tmp_path / "run")
    assert results[0].status == "ok" and results[0].conformer_search == "crest"
    [call] = fake_crest()
    conf = next(s for s in stages_of(run_dir, "PFBA") if s["stage"] == "conformers")
    assert conf["engine"] == "crest gfnff quick" and conf["threads_given"] == call["T"] == 2
    assert conf["programs"] == 1 and conf["program_wall_s"] > 0.25
    if sys.platform != "win32":                      # the fake CREST is a .bat launcher there: no CPU time
        assert conf["child_cpu_s"] >= 0.25
    [a] = rows(run_dir, "telemetry_attempts")
    assert a["crest_threads"] == 2 and a["threads_borrowed"] == 0 and a["predicted_cost"] > 0
    assert a["predicted_s"] is None                  # the built-in cost model has no scale
    [run] = rows(run_dir, "telemetry_runs")
    assert run["crest_level"] == "gfnff quick" and json.loads(run["cost_model"])["source"] == "built-in"
    assert json.loads(run["engines_json"])["programs"]["crest"]["version"] == "3.0.2"


# ----------------------------------------------------------------------------------------- lanes
def test_a_lanes_run_records_lanes_and_borrowed_crest_threads(tmp_path, fake_crest, monkeypatch):
    """3 lanes x 1 thread, 2 molecules: PFHpA's CREST borrows the unused lane's thread (-T 2); the stage row says
    so (planned 1, borrowed 1, given 2 = the -T the program got), from the lane process, written by the parent."""
    monkeypatch.setenv("FAKE_CREST_SLEEP", "0.6")
    inp = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA")]
    results, run_dir = jobs.run_resumable(inp, crest_cfg(tmp_path), tmp_path / "run", lanes=three_lanes())
    assert all(r.status == "ok" for r in results), [r.error for r in results]
    calls = {c["atoms"]: c for c in fake_crest()}
    [run] = rows(run_dir, "telemetry_runs")
    assert (run["n_lanes"], run["n_lanes_used"], run["dynamic_threads"], run["dynamic_budget"]) == (3, 2, 1, 3)
    assert [x["threads"] for x in json.loads(run["lanes_json"])] == [1, 1, 1] and run["thread_budget"] == 3
    attempts = {a["mol_id"]: a for a in rows(run_dir, "telemetry_attempts")}
    big = attempts["PFHpA"]
    assert big["lane"].startswith("cpu") and big["pid"] != os.getpid() and big["dispatch_k"] == 0   # largest first
    assert (big["crest_threads"], big["threads_borrowed"], big["size"]) == (2, 1, 23.0)
    conf = next(s for s in stages_of(run_dir, "PFHpA") if s["stage"] == "conformers")
    assert (conf["threads_planned"], conf["threads_borrowed"], conf["threads_given"]) == (1, 1, calls[23]["T"])
    assert conf["t_start_unix"] <= calls[23]["t0"] and conf["t_end_unix"] >= calls[23]["t1"]
    assert conf["predicted_cost"] is not None
    refine = next(s for s in stages_of(run_dir, "PFHpA") if s["stage"] == "refine")
    assert refine["threads_given"] == 1 and refine["threads_borrowed"] == 0       # back on the lane's own thread
    assert attempts["ethanol"]["queue_wait_s"] >= 0 and attempts["ethanol"]["n_atoms"] == 9


def test_a_lane_that_dies_before_its_first_stage_gets_a_row(tmp_path, monkeypatch):
    monkeypatch.setenv(L.TEST_CRASH_ENV, "TFE")
    results, run_dir = jobs.run_resumable(inputs(), cfg(tmp_path), tmp_path / "run",
                                          lanes=[Lane("cpu1", "cpu", 1), Lane("cpu2", "cpu", 1)])
    assert {r.input.id: r.status for r in results}["TFE"] == "failed"
    tfe = sorted(rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'TFE'"), key=lambda a: a["attempt"])
    assert [(a["attempt"], a["outcome"]) for a in tfe] == [(1, "crashed"), (2, "crashed")]
    assert "died (exit code 86)" in tfe[0]["error"] and tfe[1]["queue_wait_s"] >= 0
    assert stages_of(run_dir, "TFE") == []                 # it never reached a stage


@pytest.mark.skipif(not POSIX, reason="the fake CREST kills its lane with SIGKILL")
def test_a_lane_killed_during_crest_leaves_a_partial_row(tmp_path, fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_KILL_LANE", "23")
    inp = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA")]
    results, run_dir = jobs.run_resumable(inp, crest_cfg(tmp_path), tmp_path / "run",
                                          lanes=[Lane("cpu1", "cpu", 1), Lane("cpu2", "cpu", 1)])
    by_id = {r.input.id: r for r in results}
    assert by_id["ethanol"].status == "ok" and by_id["PFHpA"].error == "crashed the process 2 times"
    tries = sorted(rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'PFHpA'"), key=lambda a: a["attempt"])
    assert [(a["attempt"], a["outcome"], a["last_stage"]) for a in tries] == [(1, "crashed", "conformers"),
                                                                               (2, "crashed", "conformers")]
    for attempt in (1, 2):
        [s] = stages_of(run_dir, "PFHpA", attempt)
        assert s["stage"] == "conformers" and s["outcome"] == "crashed" and s["wall_s"] >= 0
        assert s["threads_given"] == tries[attempt - 1]["crest_threads"] and s["cpu_s"] is None


def test_a_lane_past_the_time_limit_records_a_timeout(tmp_path, fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_SLEEP", "30")
    c = crest_cfg(tmp_path, molecule_timeout_s=2.0)
    results, run_dir = jobs.run_resumable([from_smiles(PFBA, id="PFBA"), from_smiles(ETHANOL, id="ethanol")], c,
                                          tmp_path / "run", lanes=[Lane("cpu1", "cpu", 1), Lane("cpu2", "cpu", 1)])
    assert all(r.status == "failed" and r.error.startswith("time limit") for r in results)
    for a in rows(run_dir, "telemetry_attempts"):
        assert a["outcome"] == "timeout" and a["last_stage"] == "conformers" and a["wall_s"] >= 2.0
        [s] = stages_of(run_dir, a["mol_id"])
        assert s["outcome"] == "timeout" and s["stage"] == "conformers"


def test_a_time_limit_in_one_process_records_a_timeout(tmp_path, fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_SLEEP", "30")
    results, run_dir = jobs.run_resumable([from_smiles(PFBA, id="PFBA")], crest_cfg(tmp_path, molecule_timeout_s=1.0),
                                          tmp_path / "run")
    assert results[0].error.startswith("time limit")
    [a] = rows(run_dir, "telemetry_attempts")
    assert a["outcome"] == "timeout" and a["last_stage"] == "conformers"
    conf = stages_of(run_dir, "PFBA")[-1]
    assert conf["stage"] == "conformers" and conf["outcome"] == "timeout" and conf["programs"] == 1


# ---------------------------------------------------------------------------------- load sampler
def test_the_load_sampler_samples_and_stops():
    s = telemetry.LoadSampler(interval=0.15, first=0.05)
    s.start()
    threading.Event().wait(0.5)
    s.stop()
    got = s.take()
    assert len(got) >= 2 and not s.is_alive()
    assert all(0 <= x["cpu_percent"] <= 100 and x["mem_percent"] > 0 and x["t_rel_s"] >= 0 for x in got)
    if hasattr(os, "getloadavg"):
        assert all(x["load1"] >= 0 for x in got)
    assert s.take() == []


# ------------------------------------------------------------------------------------ aggregator
def test_timings_collects_runs_and_grid_parts(tmp_path, monkeypatch, capsys):
    """Two runs on this machine, one on another machine as a grid part (downloaded), the same run also as its
    synced folder (read once), and a job table from before the timings (counted, skipped)."""
    root = tmp_path / "Runs"
    jobs.run_resumable(inputs(["ethanol", "TFE"]), cfg(tmp_path), root / "2026-11-08_a")
    jobs.run_resumable(inputs(["acetic_acid"]), cfg(tmp_path), root / "2026-11-08_b",
                       lanes=[Lane("cpu1", "cpu", 1), Lane("cpu2", "cpu", 1)])
    other = {**telemetry.machine_info(), "machine": "other-box", "cpu_model": "Other CPU"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(telemetry, "machine_info", lambda: other)
        jobs.run_resumable(inputs(["fluoroform", "ethanol"]), cfg(tmp_path), root / "2026-11-08_c_other-box")
    part = root / "2026-11-08_grid_x" / "parts" / "other-box" / "2026-11-08_c_other-box"
    shutil.copytree(root / "2026-11-08_c_other-box", part)
    old_run_folder(root, cfg(tmp_path), inputs(["ethanol"]))
    found = timings.find_job_tables([root])
    assert len(found) == 5 and part in found
    got = timings.read_rows(found)
    assert len(got.runs) == 3 and got.duplicates == 1 and len(got.without) == 1
    assert {r["machine"] for r in got.rows} == {"other-box", telemetry.machine_info()["machine"]}
    assert {r["mol_id"] for r in got.rows if r["machine"] == "other-box"} == {"fluoroform", "ethanol"}
    assert all(r["stage"] for r in got.rows) and len(got.rows) == 5 * 5          # 5 attempts x 5 stages
    assert all(r["cpu_model"] == "Other CPU" for r in got.rows if r["machine"] == "other-box")
    one = next(r for r in got.rows if r["stage"] == "conformers")
    assert one["cpu_total_s"] is not None and one["threads_given"] == 1 and one["run_started"]
    summary = timings.summarize(got.rows)
    assert {(s["machine"], s["stage"]) for s in summary} >= {("other-box", "conformers"), ("other-box", "refine")}
    conf = next(s for s in summary if s["machine"] == "other-box" and s["stage"] == "conformers")
    assert conf["molecules"] == 2 and conf["p"] == 1.0 and conf["median_s"] > 0

    from quantabench import cli

    out_csv = tmp_path / "t.csv"
    assert cli.main(["timings", str(root), "-o", str(out_csv)]) == 0
    text = capsys.readouterr().out
    assert "3 run(s) with timings" in text and "1 run folder(s) from before the timings" in text
    assert "other-box" in text and "rdkit etkdg + MMFF94s" in text
    header = out_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(timings.CSV_FIELDS)
    assert len(out_csv.read_text(encoding="utf-8").splitlines()) == 1 + 25


def test_timings_defaults_to_the_output_folder(tmp_path, capsys):
    s = settings_mod.Settings()
    s.output_dir = str(tmp_path / "out")
    settings_mod.save(s)
    jobs.run_resumable(inputs(["ethanol"]), cfg(tmp_path), tmp_path / "out" / "run1")
    from quantabench import cli

    assert cli.main(["timings"]) == 0
    assert (tmp_path / "out" / "timings.csv").is_file()
    assert "1 run(s) with timings" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- fit
def synthetic(machine="synthbox", level="gfnff quick", n=40, widths=(2, 4, 8), p=3.7, q=0.5, a=2e-3, sigma=0.2,
              seed=7):
    rng = random.Random(seed)  # noqa: S311 - a test's reproducible draws
    out = []
    for i in range(n):
        size = rng.randint(10, 70)
        w = widths[i % len(widths)]
        t = a * size ** p * w ** -q * math.exp(rng.gauss(0.0, sigma))
        out.append({"machine": machine, "crest_level": level, "stage": "conformers", "engine": f"crest {level}",
                    "outcome": "ok", "mol_id": f"m{i}", "run_id": f"r{i % 3}", "attempt": 1, "size": float(size),
                    "threads_given": w, "wall_s": t})
    return out


def test_the_fit_recovers_p_and_q_and_runs_use_it(tmp_path, monkeypatch):
    rows_ = synthetic()
    [fit] = timings.fit_all(rows_)
    assert fit.ok, fit.why
    assert fit.power == pytest.approx(3.7, abs=0.2) and fit.speedup_power == pytest.approx(0.5, abs=0.15)
    assert fit.scatter == pytest.approx(math.exp(0.2), rel=0.25) and fit.builtin_scatter > fit.scatter
    assert fit.n_molecules == 40 and fit.widths == [2, 4, 8] and fit.power_se < 0.2
    path = timings.save_fit(fit)
    assert path == T.cost_model_path("synthbox") and path.parent.name == T.COST_MODEL_DIR
    assert str(path).startswith(str(settings_mod.settings_dir()))
    model = T.load_cost_model("synthbox", "gfnff quick")
    assert model.power == pytest.approx(fit.power) and model.source.startswith("fit synthbox gfnff quick (40 molecules")
    assert model.seconds(30, 4) == pytest.approx(fit.scale_s * 30 ** fit.power * 4 ** -fit.speedup_power)
    assert T.load_cost_model("synthbox", "gfn2 quick") is T.BUILT_IN and T.load_cost_model("nobody", "x") is T.BUILT_IN
    # the registry still reads only machine profiles (the fits are in a subfolder)
    from quantabench.dist import profile

    assert all(p.name != "synthbox" for p in profile.load_profiles())
    # a second level keeps the first
    [fit2] = timings.fit_all(synthetic(level="gfn2 quick", p=4.4))
    timings.save_fit(fit2)
    assert set(T.read_cost_models("synthbox")["models"]) == {"gfnff quick", "gfn2 quick"}


def test_the_fit_says_why_without_enough_data(tmp_path, capsys):
    [few] = timings.fit_all(synthetic(n=10))
    assert not few.ok and "10 molecule(s), needs 30" in few.why and few.power is None
    [one_width] = timings.fit_all(synthetic(widths=(2,)))
    assert not one_width.ok and "1 thread width(s) (2), needs 2" in one_width.why
    assert one_width.power_only == pytest.approx(3.7, abs=0.25)          # shown, not saved
    assert any("p alone" in line for line in timings.fit_lines(one_width))
    [steep] = timings.fit_all(synthetic(p=9.5))               # outside what the policy accepts
    assert not steep.ok and steep.power == pytest.approx(9.5, abs=0.3) and "power 9." in steep.why
    assert timings.fit_cost_model([], "x", "y").why == "no CREST searches recorded"
    assert not T.cost_model_path("synthbox").exists()


def write_synthetic_run(run_dir, rows_):
    """A job table holding only telemetry rows (CREST searches of ``rows_``), as a run on another machine left it."""
    run_dir.mkdir(parents=True)
    with jobs.JobStore(run_dir) as store, store.conn:
        machine, level = rows_[0]["machine"], rows_[0]["crest_level"]
        telemetry._insert(store.conn, "telemetry_runs", {"run_id": f"syn-{run_dir.name}", "machine": machine,
                                                          "crest_level": level, "started": "2026-11-08T00:00:00"})
        for r in rows_:
            key = {"run_id": f"syn-{run_dir.name}", "mol_id": r["mol_id"], "attempt": 1}
            telemetry._insert(store.conn, "telemetry_attempts", {**key, "outcome": "ok", "size": r["size"],
                                                                  "lane_threads": 2})
            telemetry._insert(store.conn, "telemetry_stages", {**key, "stage": "conformers", "engine": r["engine"],
                                                                "outcome": "ok", "wall_s": r["wall_s"],
                                                                "threads_given": r["threads_given"]})


def test_the_fit_cli_saves_a_fit_with_enough_data(tmp_path, capsys):
    from quantabench import cli

    data = synthetic()
    write_synthetic_run(tmp_path / "Runs" / "a", data[:20])
    write_synthetic_run(tmp_path / "Runs" / "b", data[20:])
    assert cli.main(["timings", str(tmp_path / "Runs"), "--fit", "--no-save"]) == 0
    text = capsys.readouterr().out
    assert "synthbox / CREST gfnff quick: 40 search(es) of 40 molecule(s) in 2 run(s); threads 2, 4, 8" in text
    assert "built-in: atoms^4, (threads/lane)^0.6" in text and not T.cost_model_path("synthbox").exists()
    assert cli.main(["timings", str(tmp_path / "Runs"), "--fit"]) == 0
    assert "the dynamic CREST threads of synthbox use it from now on" in capsys.readouterr().out
    assert T.load_cost_model("synthbox", "gfnff quick").power == pytest.approx(3.7, abs=0.2)
    # too few molecules for a fit: it says why and saves nothing
    write_synthetic_run(tmp_path / "few" / "a", synthetic(machine="smallbox", n=8))
    assert cli.main(["timings", str(tmp_path / "few"), "--fit"]) == 0
    assert "not saved: not enough data: 8 molecule(s), needs 30" in capsys.readouterr().out
    assert not T.cost_model_path("smallbox").exists()


def test_bad_fitted_constants_are_not_used(tmp_path):
    path = T.cost_model_path("weird")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"machine": "weird", "models": {"gfnff quick": {"power": 12.0, "speedup_power": 0.5},
                                                                "gfn2 quick": "nonsense"}}), encoding="utf-8")
    logs = []
    assert T.load_cost_model("weird", "gfnff quick", logs.append) is T.BUILT_IN
    assert logs and "power 12 outside" in logs[0]
    assert T.load_cost_model("weird", "gfn2 quick") is T.BUILT_IN
    path.write_text("{not json", encoding="utf-8")
    assert T.load_cost_model("weird", "gfnff quick") is T.BUILT_IN


def test_a_saved_fit_drives_the_dynamic_threads(tmp_path, fake_crest, monkeypatch):
    from quantabench.dist import profile

    [fit] = timings.fit_all(synthetic(machine=profile.machine_name()))
    timings.save_fit(fit)
    logs = []
    inp = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA")]
    results, run_dir = jobs.run_resumable(inp, crest_cfg(tmp_path), tmp_path / "run", lanes=three_lanes(),
                                          log=logs.append)
    assert all(r.status == "ok" for r in results)
    assert any("cost model: fit " in m and "quantabench-cli timings --fit" in m for m in logs)
    [run] = rows(run_dir, "telemetry_runs")
    assert json.loads(run["cost_model"])["power"] == pytest.approx(fit.power)
    big = rows(run_dir, "telemetry_attempts", "WHERE mol_id = 'PFHpA'")[0]
    assert big["predicted_s"] == pytest.approx(fit.scale_s * 23 ** fit.power * big["crest_threads"] ** -fit.speedup_power)


def test_timings_without_recorded_runs(tmp_path, capsys):
    from quantabench import cli

    root = tmp_path / "Runs"
    root.mkdir()
    sqlite3.connect(root / "other.sqlite").close()   # not a job table: ignored
    assert cli.main(["timings", str(root)]) == 1
    assert "0 run folder(s) with a job table, 0 run(s) with timings" in capsys.readouterr().out


def test_speedup_power_reaches_the_policy():
    """A machine whose CREST scales better (q 0.9) widens a dominant molecule further than the built-in 0.6."""
    big, other = T.crest_cost(65), T.crest_cost(59)
    assert T.choose_width(big, 4, 12, 12, rest=other * 4) == 8
    d = T.DynamicThreads(12, lambda m: {"big": big, "mid": other}[m], speedup_power=0.9)
    assert d.power == 0.9 and d.width(4, "big", ["mid"], now=0.0) >= 8
    assert d.predicted_seconds("big", 4, 8) is None
