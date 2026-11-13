"""Scratch folders (chem.external): owner records, what a failed job keeps, the sweep of what ended processes left,
and the lane kills that call it (2026-11-13: lanes killed at the time limit left 479 GB of Psi4 DF integrals)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from quantabench import settings as settings_mod
from quantabench.chem import external


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(external.SCRATCH_ENV, str(tmp_path / "scratch"))
    return external.scratch_root()


def _dead_pid() -> tuple[int, float]:
    """The pid and start time of a process that has ended."""
    import psutil

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    started = psutil.Process(p.pid).create_time()
    p.wait()
    return p.pid, started


def _left_behind(root: Path, name: str, pid: int, started: float, *, host: str | None = None,
                 age_days: float = 0.0) -> Path:
    d = root / name
    d.mkdir()
    (d / external.OWNER_FILE).write_text(json.dumps({"pid": pid, "started": started,
                                                     "host": host or socket.gethostname()}), encoding="utf-8")
    (d / f"psi.{pid}.97").write_bytes(b"\0" * 1000)                   # Psi4's DF integrals: always go
    (d / "big.bin").write_bytes(b"\0" * (external.KEEP_FAILED_MAX_BYTES + 1))     # bulky: goes
    (d / "output.dat").write_text("psi4 log", encoding="utf-8")       # the log: stays
    if age_days:
        t = time.time() - age_days * 86400
        for p in [*d.iterdir(), d]:
            os.utime(p, (t, t))
    return d


def test_scratch_root_order(tmp_path, monkeypatch):
    monkeypatch.setenv(external.SCRATCH_ENV, str(tmp_path / "env"))
    assert external.scratch_root() == tmp_path / "env"
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "set")
    settings_mod.save(s)
    assert external.scratch_root() == tmp_path / "set"          # the setting wins


def test_work_dir_has_an_owner_record_and_goes_on_success(root):
    with external.work_dir("psi4_x") as d:
        info = json.loads((d / external.OWNER_FILE).read_text(encoding="utf-8"))
        assert info["pid"] == os.getpid() and info["host"] == socket.gethostname() and info["started"] > 0
        (d / "psi.123.97").write_bytes(b"\0" * 100)
    assert not d.exists()


def test_a_failed_job_keeps_its_logs_but_not_its_scratch(root):
    with pytest.raises(external.ExternalProgramError) as exc, external.work_dir("psi4_x") as d:
        (d / "psi.123.97").write_bytes(b"\0" * 100)
        (d / "psi.123.clean").write_text("x", encoding="utf-8")
        (d / "huge.dat").write_bytes(b"\0" * (external.KEEP_FAILED_MAX_BYTES + 1))
        (d / "output.dat").write_text("log", encoding="utf-8")
        (d / "task.json").write_text("{}", encoding="utf-8")
        raise external.ExternalProgramError("psi4 failed")
    assert f"job files kept in {d}" in str(exc.value)
    assert sorted(p.name for p in d.iterdir()) == sorted([external.OWNER_FILE, "output.dat", "task.json"])
    # any other exception (a bug, a KeyboardInterrupt) prunes too
    with pytest.raises(RuntimeError), external.work_dir("xtb_y") as d2:
        (d2 / "psi.9.35").write_bytes(b"\0")
        raise RuntimeError("boom")
    assert d2.exists() and not (d2 / "psi.9.35").exists()


def test_keep_scratch_keeps_everything(root):
    s = settings_mod.Settings()
    s.engines.keep_scratch = True
    settings_mod.save(s)
    with pytest.raises(external.ExternalProgramError), external.work_dir("psi4_x") as d:
        (d / "psi.123.97").write_bytes(b"\0")
        raise external.ExternalProgramError("x")
    assert (d / "psi.123.97").exists()
    pid, started = _dead_pid()
    _left_behind(root, "psi4_dead", pid, started)
    assert external.sweep_scratch().folders == 0                  # the sweep is off too


def test_sweep_prunes_dead_owners_removes_old_ones_and_leaves_running_ones(root):
    pid, started = _dead_pid()
    dead = _left_behind(root, "psi4_dead_x", pid, started)
    old = _left_behind(root, "psi4_dead_old", pid, started, age_days=external.SCRATCH_KEEP_DAYS + 1)
    mine = _left_behind(root, "psi4_running", os.getpid(), external._this_process_started())
    other = _left_behind(root, "psi4_other_host", pid, started, host="some-other-machine")
    reused = _left_behind(root, "psi4_pid_reused", os.getpid(), 12345.0)      # this pid, another start time
    fresh = root / "crest_unowned_new"
    fresh.mkdir()
    (fresh / "psi.1.97").write_bytes(b"\0")
    stale = root / "crest_unowned_old"
    stale.mkdir()
    (stale / "psi.1.97").write_bytes(b"\0")
    t = time.time() - (external.UNOWNED_GRACE_H + 1) * 3600
    for p in [stale / "psi.1.97", stale]:
        os.utime(p, (t, t))

    dry = external.sweep_scratch(dry_run=True)
    assert (dead / f"psi.{pid}.97").exists() and old.exists()                  # nothing changed
    res = external.sweep_scratch()
    assert (res.folders, res.removed, res.pruned, res.running) == (dry.folders, dry.removed, dry.pruned, 2) == (4, 1, 3, 2)
    assert res.freed_bytes >= external.KEEP_FAILED_MAX_BYTES * 2
    assert sorted(p.name for p in dead.iterdir()) == sorted([external.OWNER_FILE, "output.dat"])
    assert not old.exists()
    for d in (mine, other):                                                    # running, or not this machine's
        owner = json.loads((d / external.OWNER_FILE).read_text(encoding="utf-8"))["pid"]
        assert (d / "big.bin").exists() and (d / f"psi.{owner}.97").exists()
    assert not (reused / "big.bin").exists()                                   # its owner ended (pid reused)
    assert (fresh / "psi.1.97").exists() and not (stale / "psi.1.97").exists()
    assert "GB freed in 4 left-behind scratch folder(s) (1 removed, 3 kept with their logs)" in res.summary()


def test_sweep_of_given_owners_only(root):
    pid, started = _dead_pid()
    pid2, started2 = _dead_pid()
    a = _left_behind(root, "psi4_a", pid, started)
    b = _left_behind(root, "psi4_b", pid2, started2)
    res = external.sweep_scratch(owners={pid})
    assert res.folders == 1 and not (a / "big.bin").exists() and (b / "big.bin").exists()


def test_remove_if_empty(root):
    d = external.make_scratch_dir("qm_x")
    (d / "density.cube").write_text("x", encoding="utf-8")
    assert not external.remove_if_empty(d) and d.exists()
    (d / "density.cube").unlink()
    assert external.remove_if_empty(d) and not d.exists()


def test_runs_sweep_when_they_start(root, tmp_path):
    from quantabench.chem import jobs
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig

    pid, started = _dead_pid()
    _left_behind(root, "psi4_left", pid, started)
    logs: list[str] = []
    cfg = BuildConfig(method="mmff", max_conformers=2, output_dir=str(tmp_path / "out"), protonate=False)
    results, _ = jobs.run_resumable([from_smiles("CCO", id="e")], cfg, tmp_path / "run", log=logs.append)
    assert results[0].status == "ok"
    assert not (root / "psi4_left" / "big.bin").exists()
    assert any(m.startswith("Scratch left by ended runs: 0.0 GB freed in 1 left-behind") for m in logs), logs


def test_cli_scratch(root, capsys):
    from quantabench import cli

    pid, started = _dead_pid()
    d = _left_behind(root, "psi4_left", pid, started)
    assert cli.main(["scratch"]) == 0
    out = capsys.readouterr().out
    assert f"Scratch folder: {root}" in out and "left behind" in out and "--clean" in out
    assert cli.main(["scratch", "--clean", "--dry-run"]) == 0
    assert "Would free 0.01 GB: 1 left-behind folder(s), 0 to remove whole, 1 kept" in capsys.readouterr().out
    assert (d / "big.bin").exists()
    assert cli.main(["scratch", "--clean", "--older-than", "0"]) == 0
    assert "1 removed whole" in capsys.readouterr().out and not d.exists()
