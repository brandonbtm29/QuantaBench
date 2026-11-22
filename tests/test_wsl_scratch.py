"""CREST through WSL in a folder inside WSL (chem.wsl_scratch, CREST-SCRATCH-1): the choice between /dev/shm and WSL's
disk, what comes back, the clean-up on success, failure, cancel and time limit, the sweeps, and "off".

The shell steps are the real scripts: a stand-in ``wsl.exe`` on PATH runs them on Linux (``wsl.exe --cd DIR -e cmd...``
becomes ``cd DIR; cmd...``), a stand-in ``df`` reports the free space of /dev/shm, and /dev/shm, WSL's home and
/proc/meminfo point into the test's folder. No real WSL is started."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_crest import FAKE_CREST, KCAL, _calls, _ethanol, _fake_program

from quantabench import settings as settings_mod
from quantabench.chem import crest, external, wsl_scratch
from quantabench.chem.optimize import Cancelled
from quantabench.chem.pipeline import BuildConfig

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="runs the WSL shell steps with a POSIX shell")
SHM = wsl_scratch.SHM_ROOT                       # /dev/shm/quantabench inside WSL
GB_KB = 1e9 / 1024                              # 1 GB in the KiB that df and /proc/meminfo report


def _set(**engines) -> None:
    s = settings_mod.load()
    for k, v in engines.items():
        setattr(s.engines, k, v)
    settings_mod.save(s)


def _dead_pid() -> tuple[int, float]:
    import psutil

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    started = psutil.Process(p.pid).create_time()
    p.wait()
    return p.pid, started


class FakeWsl:
    """The stand-in WSL: its /dev/shm root, its disk root, and the free space df and /proc/meminfo report."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.shm_root = tmp_path / "wsl" / "dev" / "shm" / "quantabench"
        self.disk_root = tmp_path / "wsl" / "home" / ".cache" / "quantabench" / "scratch"
        self.meminfo = tmp_path / "wsl" / "meminfo"
        self.meminfo.parent.mkdir(parents=True)
        self.free(shm_gb=8, mem_gb=16)
        monkeypatch.setattr(wsl_scratch, "SHM_ROOT", str(self.shm_root))
        monkeypatch.setattr(wsl_scratch, "MEMINFO", str(self.meminfo))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "wsl" / "home" / ".cache"))

    def free(self, shm_gb: float, mem_gb: float) -> None:
        self.monkeypatch.setenv("FAKE_DF_KB", str(int(shm_gb * GB_KB)))
        self.meminfo.write_text(f"MemTotal:       32000000 kB\nMemAvailable:   {int(mem_gb * GB_KB)} kB\n")

    def jobs(self) -> list[Path]:
        return sorted(p for root in (self.shm_root, self.disk_root) if root.is_dir() for p in root.iterdir())


@pytest.fixture
def fake_wsl(tmp_path, monkeypatch):
    bindir = tmp_path / "wslbin"
    bindir.mkdir()
    wsl = bindir / "wsl.exe"
    wsl.write_text('#!/bin/sh\n[ "$1" = "--cd" ] && { cd "$2" || exit 1; shift 2; }\n[ "$1" = "-e" ] && shift\n'
                   'exec "$@"\n')
    df = bindir / "df"
    df.write_text('#!/bin/sh\necho "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
                  'echo "tmpfs 16000000 0 ${FAKE_DF_KB:-0} 0% /dev/shm"\n')
    for p in (wsl, df):
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return FakeWsl(tmp_path, monkeypatch)


@pytest.fixture
def wsl_crest(tmp_path, monkeypatch, fake_wsl):
    """CREST configured as ``wsl:<fake CREST>`` (the fake writes its conformers in the folder it runs in)."""
    exe = _fake_program(tmp_path / "crestbin", "crest", FAKE_CREST)
    log = tmp_path / "crest_calls"
    monkeypatch.setenv("FAKE_CREST_LOG", str(log))
    real_find = external.find_program

    def find(name, **kw):
        if name == "crest":
            return external.ProgramInfo("crest", f"wsl:{exe}", "settings", "3.0.2")
        return real_find(name, **kw)

    monkeypatch.setattr(external, "find_program", find)
    return {"calls": lambda: _calls(log), "wsl": fake_wsl, "scratch": external.scratch_root()}


def _cfg(**kw) -> BuildConfig:
    return BuildConfig(**{"crest_method": "gfnff", "crest_quick": True, "max_conformers": 10, **kw})


# ------------------------------------------------------------------------------------- the pieces, on any OS
def test_choose_shm_when_it_and_wsl_memory_have_room():
    expected, min_b = 500_000_000, 2_000_000_000
    need = wsl_scratch.need_bytes(expected, min_b)
    assert need == 3_000_000_000                                       # the minimum + 2x the job

    def pr(shm, mem):
        return wsl_scratch.Probe("/home/u/.cache/quantabench/scratch", shm, mem)

    kind, why = wsl_scratch.choose("auto", pr(need, need), expected, min_b)
    assert kind == "shm" and "3.0 GB free" in why and "3.0 GB wanted" in why
    assert wsl_scratch.choose("auto", pr(need - 1, 64e9), expected, min_b)[0] == "disk"   # /dev/shm too small
    assert wsl_scratch.choose("auto", pr(64e9, need - 1), expected, min_b)[0] == "disk"   # WSL's memory too small
    assert wsl_scratch.choose("auto", pr(None, 64e9), expected, min_b)[0] == "disk"       # unknown: the disk
    assert wsl_scratch.choose("shm", pr(0, 0), expected, min_b) == ("shm", "crest_wsl_scratch = shm")
    assert wsl_scratch.choose("disk", pr(64e9, 64e9), expected, min_b) == ("disk", "crest_wsl_scratch = disk")
    assert crest.scratch_bytes(100) == 464 * 2**20                    # 64 MiB + 4 MiB per atom


def test_settings_modes_and_minimum():
    assert wsl_scratch.mode_setting() == "auto" and wsl_scratch.shm_min_bytes() == 2_000_000_000
    _set(crest_wsl_scratch=" SHM ", crest_wsl_shm_min_gb=0.5)
    assert wsl_scratch.mode_setting() == "shm" and wsl_scratch.shm_min_bytes() == 500_000_000
    _set(crest_wsl_scratch="tmpfs", crest_wsl_shm_min_gb=-1.0)
    assert wsl_scratch.mode_setting() == "auto" and wsl_scratch.shm_min_bytes() == 0


def test_parse_probe():
    owner = json.dumps({"pid": 4242, "started": 1.5, "host": "pc"})
    text = ("disk_root=/home/u/.cache/quantabench/scratch\nshm_free_kb=7900000\nmem_available_kb=\n"
            f"job\t/dev/shm/quantabench/crest_a_x1\t12\t2048\t{owner}\n"
            "job\t/home/u/.cache/quantabench/scratch/crest_b_x2\t90000\t4\t\n"
            "job\t/dev/shm/quantabench/crest_c_x3\t5\t8\tnot json\n")
    pr = wsl_scratch.parse_probe(text)
    assert pr.disk_root == "/home/u/.cache/quantabench/scratch"
    assert pr.shm_free == 7900000 * 1024 and pr.mem_available is None
    assert [(j.path.rsplit("/", 1)[1], j.age_s, j.size) for j in pr.jobs] == [
        ("crest_a_x1", 12.0, 2048 * 1024), ("crest_b_x2", 90000.0, 4096), ("crest_c_x3", 5.0, 8192)]
    assert pr.jobs[0].owner == {"pid": 4242, "started": 1.5, "host": "pc"}
    assert pr.jobs[1].owner is None and pr.jobs[2].owner is None
    with pytest.raises(wsl_scratch.WslScratchError, match="no home folder"):
        wsl_scratch.parse_probe("shm_free_kb=1\n")                    # no output from WSL's shell


def test_stale_folders_are_those_of_ended_owners():
    import socket

    pid, started = _dead_pid()
    me = {"pid": os.getpid(), "started": external._this_process_started(), "host": socket.gethostname()}
    dead = {"pid": pid, "started": started, "host": socket.gethostname()}
    jobs = [wsl_scratch.JobFolder(f"{SHM}/running", 5, 0, me),
            wsl_scratch.JobFolder(f"{SHM}/dead", 5, 0, dead),
            wsl_scratch.JobFolder(f"{SHM}/other_host", 5, 0, {**dead, "host": "another-pc"}),
            wsl_scratch.JobFolder(f"{SHM}/unowned_new", 60, 0, None),
            wsl_scratch.JobFolder(f"{SHM}/unowned_old", (external.UNOWNED_GRACE_H + 1) * 3600, 0, None)]
    assert [j.path.rsplit("/", 1)[1] for j in wsl_scratch.stale(jobs)] == ["dead", "unowned_old"]
    assert [j.path.rsplit("/", 1)[1] for j in wsl_scratch.stale(jobs, owners={pid})] == ["dead"]
    assert wsl_scratch.stale(jobs, owners={12345}) == []


def test_run_sh_command_and_errors(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return subprocess.CompletedProcess(cmd, 0, "dir=/dev/shm/quantabench/x\n", "")

    monkeypatch.setattr(wsl_scratch.subprocess, "run", fake_run)
    wsl_scratch.run_sh("echo hi", "/a b", 3)
    assert seen["cmd"] == ["wsl.exe", "-e", "sh", "-c", "echo hi", "sh", "/a b", "3"]
    # each step is one command-line argument of wsl.exe, as the WSL wrapper is: one line, no comments
    for script in (wsl_scratch.PROBE_SCRIPT, wsl_scratch.PREPARE_SCRIPT, wsl_scratch.FETCH_SCRIPT,
                   wsl_scratch.REMOVE_SCRIPT):
        assert "\n" not in script and " #" not in script
    assert external.OWNER_FILE in wsl_scratch.PROBE_SCRIPT and external.OWNER_FILE in wsl_scratch.PREPARE_SCRIPT
    assert seen["kw"]["stdin"] == subprocess.DEVNULL and seen["kw"]["timeout"] == wsl_scratch.SH_TIMEOUT_S
    # wsl.exe's own messages are UTF-16
    monkeypatch.setattr(wsl_scratch.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "W\x00S\x00L\x00 \x00n\x00o\x00t\x00 \x00i\x00n\x00s\x00t\x00a\x00l\x00l\x00e\x00d\x00"))
    with pytest.raises(wsl_scratch.WslScratchError, match="exit code 1: WSL not installed"):
        wsl_scratch.run_sh("true")

    def missing(cmd, **kw):
        raise FileNotFoundError(2, "No such file", "wsl.exe")

    monkeypatch.setattr(wsl_scratch.subprocess, "run", missing)
    with pytest.raises(wsl_scratch.WslScratchError, match="wsl.exe"):
        wsl_scratch.run_sh("true")


def test_remove_refuses_what_is_not_a_job_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(wsl_scratch, "run_sh", lambda script, *args, **kw: calls.append(args))
    for bad in ("/", f"{SHM}", f"{SHM}/", "/home/u/.cache/quantabench/scratch",
                "/home/u/work", "relative/quantabench/x", f"{SHM}/../../etc"):
        with pytest.raises(wsl_scratch.WslScratchError, match="not a QuantaBench job folder"):
            wsl_scratch.remove([bad])
    wsl_scratch.remove([f"{SHM}/crest_a_x1", "/home/u/.cache/quantabench/scratch/crest_b_x2", ""])
    assert calls == [(f"{SHM}/crest_a_x1", "/home/u/.cache/quantabench/scratch/crest_b_x2")]


def test_native_crest_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(wsl_scratch, "run_sh", lambda *a, **k: pytest.fail("no WSL step for a native program"))
    native = external.ProgramInfo("crest", "/opt/crest/bin/crest", "PATH")
    with wsl_scratch.job_folder(native, tmp_path, inputs=("input.xyz",)) as place:
        assert (place.kind, place.linux_dir, place.note) == ("local", None, "")
        place.fetch(crest.RESULT_FILES)                                 # nothing to fetch


def test_off_restores_the_windows_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(wsl_scratch, "run_sh", lambda *a, **k: pytest.fail("off: no WSL step"))
    _set(crest_wsl_scratch="off")
    prog = external.ProgramInfo("crest", "wsl:crest", "settings")
    with wsl_scratch.job_folder(prog, tmp_path) as place:
        assert place.kind == "windows" and place.linux_dir is None and "crest_wsl_scratch = off" in place.note


def test_no_folder_inside_wsl_falls_back_to_the_windows_folder(monkeypatch, tmp_path):
    def broken(*a, **k):
        raise wsl_scratch.WslScratchError("wsl.exe: [WinError 2] not found")

    monkeypatch.setattr(wsl_scratch, "run_sh", broken)
    lines: list[str] = []
    prog = external.ProgramInfo("crest", "wsl:crest", "settings")
    with wsl_scratch.job_folder(prog, tmp_path, log=lines.append) as place:
        assert place.kind == "windows" and place.linux_dir is None
    assert "no folder inside WSL: wsl.exe: [WinError 2] not found" in place.note
    assert lines == [f"  WARNING: {place.note}"]


def test_a_killed_lanes_program_is_killed_in_its_wsl_folder(monkeypatch):
    """The supervisor kills what a dead lane left (``kill_recorded_programs``): through WSL, in the folder it ran in."""
    seen = []
    monkeypatch.setattr(external, "IS_WINDOWS", True)
    monkeypatch.setattr(external, "_kill_wsl_side", lambda cwd, linux_cwd=None: seen.append((cwd, linux_cwd)))
    pid, started = _dead_pid()
    external._kill_recorded({"pid": pid, "started": started, "cwd": "/scratch/crest_a", "via_wsl": True,
                             "wsl_cwd": f"{SHM}/crest_a_x1"})
    external._kill_recorded({"pid": pid, "started": started, "cwd": "/scratch/crest_b", "via_wsl": True})
    assert seen == [(Path("/scratch/crest_a"), f"{SHM}/crest_a_x1"), (Path("/scratch/crest_b"), None)]


# ------------------------------------------------------------------------------------- CREST through the stand-in WSL
@posix_only
def test_crest_runs_in_dev_shm_and_only_the_results_come_back(wsl_crest, monkeypatch):
    seen_dirs, seen_log = [], []
    real_fetch = wsl_scratch.Place.fetch

    def fetch(self, names):
        seen_dirs.append(sorted(p.name for p in self.win_dir.iterdir()))
        seen_log.append((self.win_dir / "crest.log").read_text())
        real_fetch(self, names)
        seen_dirs.append(sorted(p.name for p in self.win_dir.iterdir()))

    monkeypatch.setattr(wsl_scratch.Place, "fetch", fetch)
    lines: list[str] = []
    cs = crest.generate(_ethanol(), _cfg(), threads=2, solvent="water", log=lines.append)
    assert cs.forcefield == "CREST/gfnff" and len(cs.conf_ids) == 2
    assert cs.energies_kcal == pytest.approx([-11.31 * KCAL, -11.30 * KCAL])
    (call,) = wsl_crest["calls"]()
    wsl = wsl_crest["wsl"]
    assert Path(call["cwd"]).parent == wsl.shm_root                     # CREST ran in /dev/shm/quantabench/<job>
    assert Path(call["cwd"]).name.startswith("crest_")
    assert call["argv"][:2] == ["input.xyz", "--gfnff"] and call["threads"] == "2"
    # CREST-LOG-1: one BLAS thread per CREST thread, and CREST's output in a file of its folder inside WSL, not the
    # pipe through wsl.exe; the Windows log gets it when CREST ends
    assert call["blas_threads"] == "1" and call["stdout"] == crest.OUTPUT_FILE
    assert seen_log[0] == " CREST terminated normally.\n"
    run_line = next(ln for ln in lines if ln.startswith("  $ wsl.exe"))
    assert run_line.startswith(f"  $ wsl.exe --cd {call['cwd']} -e sh -c ")
    # before the fetch the Windows job folder has the input, CREST's output (crest.log, through the pipe) and the
    # records; the fetch adds what the pipeline reads, nothing else (no trajectories, no MTD folders)
    assert seen_dirs[0] == sorted([".qb_owner.json", external.WSL_TWIN_FILE, "crest.log", "input.xyz"])
    assert set(seen_dirs[1]) - set(seen_dirs[0]) == {"crest_conformers.xyz", "crest_best.xyz"}
    assert wsl.jobs() == []                                              # the folder inside WSL is gone
    assert not any(wsl_crest["scratch"].iterdir())                       # and so is the Windows job folder
    assert len(cs.notes) == 1 and cs.notes[0].startswith("CREST scratch: WSL /dev/shm (/dev/shm 8.0 GB free, ")


@posix_only
def test_crest_takes_wsl_disk_when_dev_shm_has_no_room(wsl_crest):
    wsl = wsl_crest["wsl"]
    wsl.free(shm_gb=2.1, mem_gb=16)                     # the 2 GB minimum, but not 2x the job's size on top
    cs = crest.generate(_ethanol(), _cfg(), threads=1)
    (call,) = wsl_crest["calls"]()
    assert Path(call["cwd"]).parent == wsl.disk_root and wsl.jobs() == []
    assert cs.notes[0].startswith("CREST scratch: WSL disk (ext4) (/dev/shm 2.1 GB free, 16.0 GB of WSL's memory")
    wsl.free(shm_gb=64, mem_gb=1)                       # WSL's memory is what is short
    crest.generate(_ethanol(), _cfg(), threads=1)
    assert Path(wsl_crest["calls"]()[-1]["cwd"]).parent == wsl.disk_root
    _set(crest_wsl_shm_min_gb=0.0)                      # a smaller minimum: /dev/shm again
    wsl.free(shm_gb=2.1, mem_gb=16)
    crest.generate(_ethanol(), _cfg(), threads=1)
    assert Path(wsl_crest["calls"]()[-1]["cwd"]).parent == wsl.shm_root
    _set(crest_wsl_scratch="disk")
    crest.generate(_ethanol(), _cfg(), threads=1)
    assert Path(wsl_crest["calls"]()[-1]["cwd"]).parent == wsl.disk_root and wsl.jobs() == []


@posix_only
def test_crest_shm_falls_back_to_the_disk_when_dev_shm_cannot_be_used(wsl_crest, tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setattr(wsl_scratch, "SHM_ROOT", str(blocker / "quantabench"))     # mkdir fails there
    cs = crest.generate(_ethanol(), _cfg(), threads=1)
    assert Path(wsl_crest["calls"]()[0]["cwd"]).parent == wsl_crest["wsl"].disk_root
    assert cs.notes[0].startswith("CREST scratch: WSL disk (ext4) (/dev/shm failed: ")


@posix_only
def test_crest_off_runs_in_the_windows_folder(wsl_crest):
    _set(crest_wsl_scratch="off")
    cs = crest.generate(_ethanol(), _cfg(), threads=1)
    (call,) = wsl_crest["calls"]()
    assert Path(call["cwd"]).parent == wsl_crest["scratch"]           # the old path: the job folder through /mnt
    assert wsl_crest["wsl"].jobs() == [] and not wsl_crest["wsl"].shm_root.exists()
    assert cs.notes == ["CREST scratch: the Windows job folder through /mnt (crest_wsl_scratch = off)"]


@posix_only
def test_a_failed_crest_brings_back_its_small_files_only(wsl_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_MODE", "macfail")
    monkeypatch.setattr(external, "KEEP_FAILED_MAX_BYTES", 1000)       # crest_dynamics.trj (5000 bytes) stays behind
    with pytest.raises(external.ExternalProgramError, match="exited with code 1") as exc:
        crest.generate(_ethanol(), _cfg(), threads=1)
    assert "Initial geometry optimization failed!" in str(exc.value)
    (kept,) = [p for p in wsl_crest["scratch"].iterdir()]
    assert f"job files kept in {kept}" in str(exc.value)
    assert sorted(p.name for p in kept.iterdir()) == sorted(                # the log, CREST's small files (its whole
        [".qb_owner.json", "crest.log", crest.OUTPUT_FILE, "crestopt.log", "input.xyz"])   # output too), the input
    assert (kept / "crestopt.log").read_text() == "optimization log\n"
    assert (kept / "crest.log").read_text() == (kept / crest.OUTPUT_FILE).read_text()     # short: all of it
    assert wsl_crest["wsl"].jobs() == []                                 # the folder inside WSL is gone


@posix_only
def test_a_chatty_crest_leaves_a_bounded_log(wsl_crest, monkeypatch):
    """CREST-LOG-1: a CREST that prints the OpenBLAS warning a million times (123 MB) and then fails. Its output stays
    inside WSL; the kept log has the banner, the warning once with its count, and why CREST stopped."""
    monkeypatch.setenv("FAKE_CREST_MODE", "chatty")
    warning = ("OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with "
               "USE_OPENMP=1 option.")
    with pytest.raises(external.ExternalProgramError, match="exited with code 1") as exc:
        crest.generate(_ethanol(), _cfg(), threads=4)
    assert "Initial geometry optimization failed!" in str(exc.value)                # the tail is in the error
    assert crest.failure_reason(str(exc.value)).startswith("Initial geometry optimization failed! (exit code 1")
    (kept,) = [p for p in wsl_crest["scratch"].iterdir()]
    assert crest.OUTPUT_FILE not in {p.name for p in kept.iterdir()}               # 123 MB: it stays behind
    text = (kept / "crest.log").read_text()
    assert len(text) < 1000 and text.count(warning) == 1
    assert text.splitlines() == [" CREST banner", warning,
                                 "    [QuantaBench: the line above repeated 999999 more times]",
                                 "  Initial geometry optimization failed!"]
    assert wsl_crest["wsl"].jobs() == []


@posix_only
def test_a_cancelled_crest_leaves_nothing(wsl_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_MODE", "sleep")
    flag = threading.Event()
    threading.Timer(1.5, flag.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        crest.generate(_ethanol(), _cfg(), threads=1, should_stop=flag.is_set)
    assert time.monotonic() - t0 < 30
    (call,) = wsl_crest["calls"]()
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(call["pid"], 0)                                          # the Linux side was killed
    assert wsl_crest["wsl"].jobs() == [] and not any(wsl_crest["scratch"].iterdir())


@posix_only
def test_a_crest_past_the_time_limit_is_cleaned_up(wsl_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_MODE", "sleep")
    _set(timeout_s=1.0)
    with pytest.raises(external.ExternalProgramError, match="timed out after 1 s"):
        crest.generate(_ethanol(), _cfg(), threads=1)
    (kept,) = [p for p in wsl_crest["scratch"].iterdir()]
    assert {"crest.log", "crest_dynamics.trj"} <= {p.name for p in kept.iterdir()}   # small enough to come back
    assert not (kept / "MTD1").exists() and wsl_crest["wsl"].jobs() == []


@posix_only
def test_keep_scratch_brings_back_everything(wsl_crest):
    _set(keep_scratch=True)
    crest.generate(_ethanol(), _cfg(), threads=1)
    (kept,) = [p for p in wsl_crest["scratch"].iterdir()]
    names = {p.name for p in kept.iterdir()}
    assert {"crest_conformers.xyz", "crest_dynamics.trj", "MTD1", "crest.log", "input.xyz"} <= names
    assert ".qb_pid" not in names and (kept / "MTD1" / "xtb.trj").read_text() == "trajectory\n"
    assert json.loads((kept / external.OWNER_FILE).read_text())["pid"] == os.getpid()
    assert wsl_crest["wsl"].jobs() == []                                 # kept in the Windows folder, not in WSL


@posix_only
def test_an_error_in_the_block_cleans_up_too(fake_wsl, tmp_path):
    prog = external.ProgramInfo("crest", "wsl:crest", "settings")
    win = tmp_path / "job"
    win.mkdir()
    (win / "input.xyz").write_text("1\n\nH 0 0 0\n")
    with pytest.raises(RuntimeError, match="boom"), wsl_scratch.job_folder(prog, win, inputs=("input.xyz",)) as place:
        linux = Path(place.linux_dir)
        assert (linux / "input.xyz").read_text() == "1\n\nH 0 0 0\n"
        assert json.loads((linux / external.OWNER_FILE).read_text())["win_dir"] == str(win)
        assert external.wsl_twin(win) == place.linux_dir
        (linux / "notes.txt").write_text("why it failed")
        (linux / ".qb_pid").write_text("123")
        raise RuntimeError("boom")
    assert (win / "notes.txt").read_text() == "why it failed" and not (win / ".qb_pid").exists()
    assert not linux.exists() and not (win / external.WSL_TWIN_FILE).exists()


# ------------------------------------------------------------------------------------- a program's output inside WSL
def _wsl_job_dir(fake_wsl) -> Path:
    d = fake_wsl.shm_root / "job_x1"
    d.mkdir(parents=True)
    return d


def _wsl_sh(script: str) -> list[str]:
    """A shell script run through the stand-in WSL with the WSL wrapper, as ``ProgramInfo.command()`` builds it."""
    return ["wsl.exe", "-e", "sh", "-c", external.WSL_WRAPPER, "sh", "-c", script]


@posix_only
def test_the_same_line_a_million_times_comes_back_once(fake_wsl, tmp_path):
    linux = _wsl_job_dir(fake_wsl)
    lines: list[str] = []
    line = "OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with " \
           "USE_OPENMP=1 option."
    run = external.run_program(_wsl_sh(f'echo start; yes "{line}" | head -n 1000000; echo "the end" >&2'),
                               tmp_path / "win", log_name="chatty.log", program_cwd=str(linux),
                               program_log="chatty.out", log=lines.append)
    assert (linux / "chatty.out").stat().st_size == 6 + 1_000_000 * (len(line) + 1) + 8    # all of it, inside WSL
    assert run.log_file.read_text().splitlines() == [
        "start", line, "    [QuantaBench: the line above repeated 999999 more times]", "the end"]
    assert [m for m in lines if "WARNING" in m] == ["  WARNING: sh wrote 116 MB of output inside WSL; the log has its "
                                                    "first and last lines, repeated lines collapsed"]


@posix_only
def test_a_long_output_comes_back_as_its_first_and_last_lines(fake_wsl, tmp_path):
    linux = _wsl_job_dir(fake_wsl)
    lines: list[str] = []
    run = external.run_program(_wsl_sh("seq 1 1000000; exit 3"), tmp_path / "win", log_name="seq.log",
                               program_cwd=str(linux), program_log="seq.out", check=False, log=lines.append)
    assert run.returncode == 3
    text = run.log_file.read_text()
    got = text.splitlines()
    n_head, n_tail = wsl_scratch.OUTPUT_HEAD_LINES, wsl_scratch.OUTPUT_TAIL_LINES
    assert got[:n_head] == [str(i) for i in range(1, n_head + 1)]
    assert got[n_head] == (f"    [QuantaBench: {1_000_000 - n_head - n_tail} lines left out here; the whole output, "
                           "seq.out inside WSL, has 1000000 lines and 6888896 bytes]")
    assert got[n_head + 1:] == [str(i) for i in range(1_000_001 - n_tail, 1_000_001)]
    assert len(text.encode()) <= wsl_scratch.OUTPUT_MAX_BYTES
    assert not [m for m in lines if "WARNING" in m]                                  # 6.9 MB: no warning
    # long lines are cut, and the bytes bound the tail when 2,000 lines would not fit
    long = "x" * 3000
    run = external.run_program(_wsl_sh(f"for i in $(seq 1 3000); do echo {long}$i; done"), tmp_path / "win2",
                               log_name="long.log", program_cwd=str(linux), program_log="long.out")
    text = run.log_file.read_text()
    assert len(text.encode()) <= wsl_scratch.OUTPUT_MAX_BYTES + 200
    got = text.splitlines()
    assert got[0] == "x" * wsl_scratch.OUTPUT_LINE_CHARS + " [QuantaBench: 1001 more characters]"
    assert "lines left out here" in text and got[-1].endswith(" [QuantaBench: 1004 more characters]")


@posix_only
def test_the_output_of_a_program_past_its_time_limit_comes_back(fake_wsl, tmp_path):
    linux = _wsl_job_dir(fake_wsl)
    with pytest.raises(external.ExternalProgramError, match="timed out after 1 s") as exc:
        external.run_program(_wsl_sh("echo working; sleep 30"), tmp_path / "win", log_name="slow.log",
                             program_cwd=str(linux), program_log="slow.out", timeout=1.0)
    assert exc.value.args[0].endswith("\nworking")                     # the error's log tail
    assert (tmp_path / "win" / "slow.log").read_text() == "working\n"


@posix_only
def test_no_output_file_without_a_folder_inside_wsl(fake_wsl, tmp_path):
    """``program_log`` is only for a program in a folder inside WSL: elsewhere the output goes through the pipe."""
    run = external.run_program(_wsl_sh("echo through the pipe"), tmp_path / "win", log_name="p.log",
                               program_log="p.out")
    assert run.text() == "through the pipe\n" and not (tmp_path / "win" / "p.out").exists()
    linux = _wsl_job_dir(fake_wsl)
    run = external.run_program(_wsl_sh("echo through the pipe"), tmp_path / "win2", log_name="p.log",
                               program_cwd=str(linux))
    assert run.text() == "through the pipe\n" and not (linux / "p.out").exists()
    assert wsl_scratch.output_tail(str(linux), "missing.out") == ("", 0)


# ------------------------------------------------------------------------------------- the sweeps
def _wsl_job(root: Path, name: str, owner: dict | None, age_h: float = 0.0) -> Path:
    d = root / name
    (d / "MTD1").mkdir(parents=True)
    (d / "MTD1" / "xtb.trj").write_bytes(b"x" * 4096)
    if owner is not None:
        (d / external.OWNER_FILE).write_text(json.dumps(owner))
    if age_h:
        t = time.time() - age_h * 3600
        os.utime(d, (t, t))
    return d


@posix_only
def test_each_job_removes_the_wsl_folders_of_ended_runs(fake_wsl, tmp_path):
    import socket

    pid, started = _dead_pid()
    host = socket.gethostname()
    root = fake_wsl.shm_root
    dead = _wsl_job(root, "crest_dead_a1", {"pid": pid, "started": started, "host": host})
    old = _wsl_job(fake_wsl.disk_root, "crest_unowned_b2", None, age_h=external.UNOWNED_GRACE_H + 1)
    new = _wsl_job(root, "crest_unowned_c3", None)
    mine = _wsl_job(root, "crest_running_d4", {"pid": os.getpid(), "started": external._this_process_started(),
                                                "host": host})
    other = _wsl_job(root, "crest_other_e5", {"pid": pid, "started": started, "host": "another-pc"})
    lines: list[str] = []
    prog = external.ProgramInfo("crest", "wsl:crest", "settings")
    with wsl_scratch.job_folder(prog, tmp_path, log=lines.append) as place:
        assert place.kind == "shm"
    assert not dead.exists() and not old.exists()
    assert new.exists() and mine.exists() and other.exists()
    assert any(ln.startswith("  CREST scratch: removed 2 job folder(s) of ended runs inside WSL") for ln in lines)
    # quantabench-cli scratch --clean: the same rule, on demand
    for d in (new, mine, other):
        os.utime(d, (time.time() - 13 * 3600,) * 2)
    res = wsl_scratch.sweep(dry_run=True)
    assert (res.folders, res.running) == (1, 2) and new.exists()        # dry run: nothing removed
    res = wsl_scratch.sweep()
    assert (res.folders, res.removed, res.running) == (1, 1, 2) and res.freed_bytes >= 4096
    assert not new.exists() and mine.exists() and other.exists()
    _set(keep_scratch=True)                                             # keep_scratch turns the sweep off
    assert wsl_scratch.sweep().folders == 0


@posix_only
def test_sweep_scratch_removes_the_wsl_twin_of_a_left_behind_folder(fake_wsl):
    import socket

    pid, started = _dead_pid()
    root = external.scratch_root()
    twins = []
    for name, owner in (("crest_dead", {"pid": pid, "started": started, "host": socket.gethostname()}),
                        ("crest_running", {"pid": os.getpid(), "started": external._this_process_started(),
                                           "host": socket.gethostname()})):
        win = root / name
        win.mkdir()
        (win / external.OWNER_FILE).write_text(json.dumps(owner))
        (win / "crest.log").write_text("CREST")
        twin = _wsl_job(fake_wsl.shm_root, f"{name}_x1", owner)
        (win / external.WSL_TWIN_FILE).write_text(json.dumps({"dir": str(twin), "kind": "shm"}))
        twins.append((win, twin))
    assert external.sweep_scratch(dry_run=True).wsl_removed == 0 and twins[0][1].exists()
    res = external.sweep_scratch(owners={pid})                           # a lane just killed
    assert res.folders == 1 and res.wsl_removed == 1 and "1 job folder(s) inside WSL removed" in res.summary()
    (dead_win, dead_twin), (run_win, run_twin) = twins
    assert not dead_twin.exists() and not (dead_win / external.WSL_TWIN_FILE).exists()
    assert (dead_win / "crest.log").exists()                             # the Windows folder keeps its log
    assert run_twin.exists() and (run_win / external.WSL_TWIN_FILE).exists()    # a running job is left alone


@posix_only
def test_cli_scratch_lists_and_cleans_inside_wsl(fake_wsl, monkeypatch, capsys):
    from quantabench import cli

    monkeypatch.setattr(wsl_scratch, "in_use", lambda: True)
    pid, started = _dead_pid()
    import socket

    _wsl_job(fake_wsl.shm_root, "crest_dead_a1", {"pid": pid, "started": started, "host": socket.gethostname()})
    assert cli.main(["scratch"]) == 0
    out = capsys.readouterr().out
    assert f"Inside WSL: {fake_wsl.shm_root} and {fake_wsl.disk_root}" in out and "left behind" in out
    assert cli.main(["scratch", "--clean"]) == 0
    assert "Inside WSL: freed 0.00 GB in 1 job folder(s) of ended runs; 0 of running jobs left alone" in \
        capsys.readouterr().out
    assert fake_wsl.jobs() == []


def test_cli_settings_sets_the_mode(capsys):
    from quantabench import cli

    assert cli.main(["settings", "--crest-wsl-scratch", "disk"]) == 0
    assert "CREST scratch through WSL: disk" in capsys.readouterr().out
    assert settings_mod.load().engines.crest_wsl_scratch == "disk"
    with pytest.raises(SystemExit):
        cli.main(["settings", "--crest-wsl-scratch", "tmpfs"])
