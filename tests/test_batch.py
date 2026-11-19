"""quantabench-cli batch (BATCH-1): phases, attempts, the deadline, snapshots of remote shares, resume, the report."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantabench import cli
from quantabench import settings as settings_mod
from quantabench.chem.inputs import from_smiles
from quantabench.chem.pipeline import BuildConfig
from quantabench.dist import batch

MOLS = [("ethanol", "CCO"), ("propanol", "CCCO"), ("butanol", "CCCCO"), ("pentanol", "CCCCCO")]


def items():
    return [from_smiles(s, id=i) for i, s in MOLS]


def write_run(folder: Path, ids, phase: str, status="ok") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "phase", "smiles"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "status": status, "phase": phase, "smiles": dict(MOLS)[i]})


# ------------------------------------------------------------------------------------------- parts
def test_deadline_parsing():
    now = datetime(2026, 11, 18, 15, 0)
    assert batch.parse_deadline("07:00", now) == datetime(2026, 11, 19, 7, 0).timestamp()      # the next one
    assert batch.parse_deadline("16:30", now) == datetime(2026, 11, 18, 16, 30).timestamp()
    assert batch.parse_deadline("2026-11-20T08:15", now) == datetime(2026, 11, 20, 8, 15).timestamp()
    assert batch.parse_deadline("none") is None and batch.parse_deadline("") is None and batch.parse_deadline(None) is None
    with pytest.raises(ValueError, match="HH:MM"):
        batch.parse_deadline("tomorrow", now)


def ref(name, reachable=True, **info):
    return SimpleNamespace(name=name, reachable=reachable, info=info, error="" if reachable else "timed out")


def test_ready_machines_asks_the_workers():
    from quantabench import __version__

    refs = {"w1": ref("w1", quantabench_version=__version__, queue={}),
            "w2": ref("w2", quantabench_version=__version__, queue={"paused": True}),
            "w3": ref("w3", quantabench_version="0.0.1", queue={}),
            "w4": ref("w4", reachable=False)}
    discover = lambda *a, **k: refs  # noqa: E731
    machines, note = batch.ready_machines("auto", discover=discover)
    assert machines == ["w1", "local"]
    assert "w2 not used: paused" in note and "w3 not used: QuantaBench 0.0.1" in note and "w4 not used: unreachable" in note
    machines, note = batch.ready_machines("auto", include_paused=True, discover=discover)
    assert machines == ["w1", "w2", "local"] and "w2 ready (paused: works once resumed)" in note
    assert batch.ready_machines("local", discover=discover) == (["local"], "this machine alone")
    machines, note = batch.ready_machines("w1,w3", discover=discover)
    assert machines is None and note.startswith("w3 is not ready: QuantaBench 0.0.1")
    machines, note = batch.ready_machines("auto", discover=lambda *a, **k: {})
    assert machines == ["local"] and "no workers known" in note


def test_snapshots_come_from_every_unmerged_grid_of_the_phase(tmp_path):
    pdir, synced = tmp_path / "gas", tmp_path / "synced"
    (pdir / "g1").mkdir(parents=True)
    (pdir / "g1" / "grid.json").write_text(json.dumps({"state": "running", "shares": [
        {"machine": "local", "run_folder": "here"}, {"machine": "mac", "run_folder": "mac_run"},
        {"machine": "pc", "run_folder": "../escape"}]}), encoding="utf-8")
    (pdir / "g2").mkdir()
    (pdir / "g2" / "grid.json").write_text(json.dumps({"state": "merged", "shares": [
        {"machine": "mac", "run_folder": "old_run"}]}), encoding="utf-8")
    assert batch.remote_run_folders(pdir) == [("mac", "mac_run")]
    write_run(synced / "mac_run", ["ethanol", "propanol"], "gas")
    lines = []
    assert batch.snapshot_remotes(pdir, synced, lines.append) == 2
    assert (pdir / "partial_mac_run" / "summary.csv").is_file() and "2 molecule(s) finished in mac_run" in lines[0]
    assert batch.finished_ids(pdir) == {"ethanol", "propanol"}
    # a crashed molecule recorded in grid.json counts as finished (failed, no row)
    (pdir / "g1" / "grid.json").write_text(json.dumps({"state": "running", "shares": [],
                                                      "failed_without_result": ["butanol"]}), encoding="utf-8")
    assert batch.finished_ids(pdir) == {"ethanol", "propanol", "butanol"}


# ------------------------------------------------------------------------------- a batch, faked grid
class FakeGrids:
    """grid_build stand-in: each call finishes ``per_call`` molecules locally (a run folder in the phase folder);
    ``stop_after`` calls in, the deadline passes during the call, half of the rest is left to a remote share whose
    synced run folder has one of them done."""

    def __init__(self, synced: Path, clock, per_call=10, deadline_on_call=None):
        self.synced, self.clock, self.per_call, self.deadline_on_call = synced, clock, per_call, deadline_on_call
        self.calls = []

    def __call__(self, todo, cfg, machines, weights, *, label, wait, output_dir, log, should_stop, **kw):
        self.calls.append({"ids": [t.id for t in todo], "machines": machines, "solvent": cfg.solvent, "label": label})
        phase = cfg.solvent or "gas"
        out = Path(output_dir)
        ids = [t.id for t in todo]
        gfolder = out / f"{label}_grid"
        gfolder.mkdir(parents=True)
        if self.deadline_on_call == len(self.calls):
            self.clock.t = 10_000.0                       # the deadline passes during this attempt
            assert should_stop()
            write_run(out / f"{label}_local", ids[:1], phase)
            write_run(self.synced / f"{label}_mac", ids[1:2], phase)
            shares = [{"machine": "local", "run_folder": f"{label}_local"},
                      {"machine": "mac", "run_folder": f"{label}_mac", "job_id": "j1", "url": "http://x"}]
            (gfolder / "grid.json").write_text(json.dumps({"state": "running", "shares": shares}), encoding="utf-8")
            return SimpleNamespace(id=gfolder.name, state="running", missing=[], shares=[])
        done = ids[: self.per_call]
        write_run(gfolder, done, phase)
        missing = ids[self.per_call:]
        (gfolder / "grid.json").write_text(json.dumps({"state": "merged", "shares": []}), encoding="utf-8")
        return SimpleNamespace(id=gfolder.name, state="merged", missing=missing, shares=[])


def test_a_batch_stops_at_its_deadline_resumes_and_pairs(tmp_path):
    clock = SimpleNamespace(t=0.0)
    synced = tmp_path / "synced"
    grids = FakeGrids(synced, clock, deadline_on_call=1)
    cancelled = []
    opts = batch.BatchOptions(phases=["", "water"], deadline=5_000.0)
    run_dir = tmp_path / "b1"
    lines = []
    kw = dict(synced_root=synced, echo=lines.append, grid_build=grids, ready=lambda: (["mac", "local"], "mac ready"),
              cancel=lambda g, _h: cancelled.append(g.id) or ["mac: job j1 cancelled"], clock=lambda: clock.t)
    cfg = BuildConfig(method="mmff")
    assert batch.run_batch(items(), cfg, run_dir, opts, **kw) == batch.EXIT_STOPPED
    assert cancelled == ["b1_gas_r0_grid"]
    assert any("STOPPED at the deadline during gas r0" in m for m in lines)
    # the local molecule and the remote one (snapshot of its synced run folder) both count
    assert batch.finished_ids(run_dir / "gas") == {"ethanol", "propanol"}
    assert (run_dir / "gas" / "partial_b1_gas_r0_mac" / "summary.csv").is_file()
    st = json.loads((run_dir / "gas" / "phase.json").read_text(encoding="utf-8"))
    assert st["attempts"][0]["stopped"] == "deadline" and not st.get("done")
    assert json.loads((run_dir / "batch.json").read_text(encoding="utf-8"))["exit"] == batch.EXIT_STOPPED
    # past the deadline, a new start stops before an attempt
    assert batch.run_batch(items(), cfg, run_dir, opts, **kw) == batch.EXIT_STOPPED
    assert len(grids.calls) == 1 and any("has passed before gas attempt 2" in m for m in lines)
    # resumed without a deadline: only the two unfinished molecules run, then water; a molecule that no machine
    # returned (per_call 1) is run by the next attempt
    grids.per_call = 1
    opts.deadline = None
    lines.clear()
    assert batch.run_batch(items(), cfg, run_dir, opts, **kw) == batch.EXIT_OK
    assert grids.calls[1]["ids"] == ["butanol", "pentanol"] and grids.calls[2]["ids"] == ["pentanol"]
    water = [c for c in grids.calls if c["solvent"] == "water"]
    assert [c["ids"] for c in water] == [[i for i, _ in MOLS], [i for i, _ in MOLS][1:], [i for i, _ in MOLS][2:],
                                         [i for i, _ in MOLS][3:]]
    assert all(c["machines"] == ["mac", "local"] for c in grids.calls)
    assert json.loads((run_dir / "water" / "phase.json").read_text(encoding="utf-8"))["done"]
    assert any(m.startswith("ALL-DONE batch b1") for m in lines)
    assert any(m.startswith("paired 4 molecule(s) of gas, water") or m.startswith("pair:") for m in lines)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| gas | r0 | mac, local | 4 |" in report and "| water | total | | 4 with a result | | done |" in report
    assert batch.status_lines(run_dir)[1].strip().startswith("gas      4/4 with a result, 3 attempt(s), done")
    # a finished batch run again does nothing
    n = len(grids.calls)
    assert batch.run_batch(items(), cfg, run_dir, opts, **kw) == batch.EXIT_OK
    assert len(grids.calls) == n and any("gas: done in an earlier session" in m for m in lines)


def test_attempts_are_limited_and_a_named_machine_must_be_ready(tmp_path):
    clock = SimpleNamespace(t=0.0)
    grids = FakeGrids(tmp_path / "synced", clock, per_call=0)            # nothing ever comes back
    lines = []
    opts = batch.BatchOptions(phases=[""], max_attempts=2)
    kw = dict(synced_root=tmp_path / "synced", echo=lines.append, grid_build=grids, clock=lambda: clock.t)
    code = batch.run_batch(items(), BuildConfig(method="mmff"), tmp_path / "b2", opts,
                           ready=lambda: (["local"], "this machine alone"), **kw)
    assert code == batch.EXIT_FAILED and len(grids.calls) == 2 and any("has had 2 attempts" in m for m in lines)
    code = batch.run_batch(items(), BuildConfig(method="mmff"), tmp_path / "b3", opts,
                           ready=lambda: (None, "mac is not ready: paused"), **kw)
    assert code == batch.EXIT_NOT_READY and any("STOP: mac is not ready: paused" in m for m in lines)
    dup = items() + [from_smiles("CO", id="ethanol")]
    assert batch.run_batch(dup, BuildConfig(method="mmff"), tmp_path / "b4", opts,
                           ready=lambda: (["local"], ""), **kw) == batch.EXIT_USAGE


def test_a_failed_attempt_is_retried(tmp_path):
    calls = []

    def flaky(todo, cfg, machines, weights, **kw):
        calls.append(len(todo))
        if len(calls) == 1:
            raise RuntimeError("no answer from the registry")
        return FakeGrids(tmp_path / "synced", None)(todo, cfg, machines, weights, **kw)

    lines = []
    code = batch.run_batch(items(), BuildConfig(method="mmff"), tmp_path / "b5", batch.BatchOptions(phases=[""]),
                           synced_root=tmp_path / "synced", echo=lines.append, grid_build=flaky,
                           ready=lambda: (["local"], ""))
    assert code == batch.EXIT_OK and calls == [4, 4]
    assert any("RuntimeError: no answer from the registry" in m for m in lines)


# ------------------------------------------------------------------------------- the CLI, a real local grid
@pytest.fixture
def local_settings(isolated_settings, monkeypatch):
    s = settings_mod.Settings(output_dir=str(isolated_settings / "runs"))
    s.resources.threads, s.resources.gpu = 1, "off"
    settings_mod.save(s)
    monkeypatch.setattr("quantabench.chem.properties.available", lambda: False)     # no dxtb step in tests
    return s


def test_batch_run_on_this_machine_and_resume(tmp_path, local_settings, capsys):
    run_dir = tmp_path / "cli_batch"
    argv = ["batch", "run", "CCO ethanol", "CCCO propanol", "--method", "mmff", "--max-confs", "2", "--refine-top",
            "1", "--phases", "gas", "--run-dir", str(run_dir), "--machines", "local", "--no-review", "--rank", "refine"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "start batch cli_batch: 2 molecule(s), phases gas, deadline none" in out and "ALL-DONE batch cli_batch" in out
    assert batch.finished_ids(run_dir / "gas") == {"ethanol", "propanol"}
    assert (run_dir / "report.md").is_file() and (run_dir / "grid_gas_r0.log").is_file()
    assert cli.main(argv) == 0
    assert "gas: done in an earlier session (1 attempt(s))" in capsys.readouterr().out
    assert cli.main(["batch", "status", str(run_dir)]) == 0
    assert "gas      2/2 with a result, 1 attempt(s), done" in capsys.readouterr().out


def test_batch_list_and_bad_options(tmp_path, local_settings, capsys):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("id,smiles\nethanol,CCO\n", encoding="utf-8")
    b.write_text("id,smiles\npropanol,CCCO\n", encoding="utf-8")
    lst = tmp_path / "batches.txt"
    lst.write_text("# two batches\na.csv\nb.csv\n", encoding="utf-8")
    root = tmp_path / "root"
    assert cli.main(["batch", "run", "--list", str(lst), "--root", str(root), "--method", "mmff", "--max-confs", "2",
                     "--refine-top", "1", "--machines", "local", "--no-review", "--rank", "refine"]) == 0
    assert batch.finished_ids(root / "a" / "gas") == {"ethanol"} and batch.finished_ids(root / "b" / "gas") == {"propanol"}
    capsys.readouterr()
    assert cli.main(["batch", "run", "CCO", "--deadline", "soon", "--method", "mmff"]) == 2
    assert "HH:MM" in capsys.readouterr().err
    assert cli.main(["batch", "run", "--list", str(lst), "--file", str(a), "--method", "mmff"]) == 2
    assert cli.main(["batch", "run", "CCO", "--phases", "gas,gas", "--method", "mmff"]) == 2
