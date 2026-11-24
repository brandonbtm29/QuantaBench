"""quantabench.local_worker: the worker on this computer as an update sees it (DIST-1).

A child process stands in for the worker (it holds a ``worker.lock`` written here); launchctl and schtasks are
fakes, so no real worker or service is ever touched.
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from quantabench import local_worker


@pytest.fixture
def fake_worker(tmp_path):
    """(state folder, repo, process): a sleeping Python started in ``repo`` that the lock names."""
    repo = tmp_path / "QuantaBench"
    repo.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=repo)
    try:
        started = psutil.Process(proc.pid).create_time()
        (state / "worker.lock").write_text(json.dumps({"pid": proc.pid, "pid_started": started,
                                                       "host": socket.gethostname(), "since": "now"}), encoding="utf-8")
        yield state, repo, proc
    finally:
        proc.kill()
        proc.wait(10)


def jobs(state: Path, *states: str) -> None:
    (state / "jobs.json").write_text(json.dumps({"format": 1, "jobs": [
        {"id": f"j{i}", "label": f"job {i}", "state": s} for i, s in enumerate(states)]}), encoding="utf-8")


def no_service(cmd, **kw):
    return subprocess.CompletedProcess(cmd, 1, "", "not found")


def test_no_lock_or_a_dead_one(tmp_path):
    assert local_worker.find(tmp_path, folder=tmp_path, run=no_service) is None
    (tmp_path / "worker.lock").write_text(json.dumps({"pid": 999999, "pid_started": 1.0,
                                                      "host": socket.gethostname()}), encoding="utf-8")
    assert local_worker.find(tmp_path, folder=tmp_path, run=no_service) is None
    (tmp_path / "worker.lock").write_text("{torn", encoding="utf-8")
    assert local_worker.find(tmp_path, folder=tmp_path, run=no_service) is None


def test_idle_worker_of_this_checkout(fake_worker):
    state, repo, proc = fake_worker
    w = local_worker.find(repo, folder=state, run=no_service)
    assert w is not None and w.pid == proc.pid and w.same_checkout is True and w.affected
    assert not w.busy and w.busy_reason() == "" and w.manager == "manual" and not w.can_restart
    assert "is idle" in w.describe() and "started by hand" in w.describe()
    assert "Ctrl+C" in w.restart_hint()
    # the same worker seen from another checkout does not matter to it
    other = local_worker.find(repo.parent / "elsewhere", folder=state, run=no_service)
    assert other.same_checkout is False and not other.affected


@pytest.mark.parametrize(("states", "paused", "busy"), [
    (("running",), False, True), (("queued",), False, True), (("queued",), True, False),
    (("running", "queued"), True, True), (("paused", "done", "failed"), False, False), ((), False, False)])
def test_busy(fake_worker, states, paused, busy):
    state, repo, _ = fake_worker
    jobs(state, *states)
    if paused:
        (state / "pause.json").write_text(json.dumps({"paused": True, "mode": "drain"}), encoding="utf-8")
    w = local_worker.find(repo, folder=state, run=no_service)
    assert w.busy is busy and bool(w.busy_reason()) is busy and w.machine_paused is paused
    if busy:
        assert "job 0" in w.busy_reason() and "pause the machine" in w.busy_reason()
    if "paused" in states:
        assert "1 paused" in w.describe()


def test_saved_job_list_backup_is_read(fake_worker):
    state, repo, _ = fake_worker
    (state / "jobs.json.bak").write_text(json.dumps({"jobs": [{"id": "a", "state": "running"}]}), encoding="utf-8")
    assert local_worker.find(repo, folder=state, run=no_service).running == ["a"]


def test_service_detection(fake_worker):
    state, repo, proc = fake_worker

    def launchctl(cmd, **kw):
        assert cmd[:2] == ["launchctl", "print"] and cmd[2].endswith(local_worker.LAUNCHD_LABEL)
        return subprocess.CompletedProcess(cmd, 0, f"gui/501/x = {{\n\tstate = running\n\tpid = {proc.pid}\n}}\n", "")

    w = local_worker.find(repo, folder=state, run=launchctl, platform="darwin")
    assert w.manager == "launchd" and w.can_restart and "LaunchAgent" in w.describe()
    assert w.restart_hint().startswith("launchctl kickstart -k gui/")

    def schtasks(cmd, **kw):
        assert cmd[:4] == ["schtasks", "/Query", "/TN", local_worker.WINDOWS_TASK]
        return subprocess.CompletedProcess(cmd, 0, '"\\QuantaBench worker","N/A","Running"\r\n', "")

    w = local_worker.find(repo, folder=state, run=schtasks, platform="win32")
    assert w.manager == "task" and w.can_restart and "Start-ScheduledTask" in w.restart_hint()
    ready = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, '"\\QuantaBench worker","N/A","Ready"\r\n', "")  # noqa: E731
    assert local_worker.find(repo, folder=state, run=ready, platform="win32").manager == "manual"
    other_pid = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "pid = 1\n", "")  # noqa: E731
    assert local_worker.find(repo, folder=state, run=other_pid, platform="darwin").manager == "manual"


def _worker(state, **kw):
    base = {"pid": 4321, "state_dir": state, "same_checkout": True, "manager": "launchd"}
    base.update(kw)
    return local_worker.LocalWorker(**base)


def test_restart_launchd(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    ok, msg = local_worker.restart(_worker(tmp_path), run=run, platform="darwin",
                                   wait_new=lambda folder, old, t: 5555)
    assert ok and "4321 -> 5555" in msg
    assert calls == [["launchctl", "kickstart", "-k", f"gui/{local_worker._uid()}/{local_worker.LAUNCHD_LABEL}"]]
    ok, msg = local_worker.restart(_worker(tmp_path), run=run, platform="darwin", wait=1,
                                   wait_new=lambda folder, old, t: None)
    assert not ok and "has not come back" in msg
    bad = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 5, "", "Could not find service")  # noqa: E731
    ok, msg = local_worker.restart(_worker(tmp_path), run=bad, platform="darwin")
    assert not ok and "Could not find service" in msg


def test_restart_scheduled_task(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[:2])
        return subprocess.CompletedProcess(cmd, 0, "SUCCESS", "")

    ok, msg = local_worker.restart(_worker(tmp_path, manager="task"), run=run, platform="win32",
                                   wait_gone=lambda pid, t: True, wait_new=lambda folder, old, t: 77)
    assert ok and calls == [["schtasks", "/End"], ["schtasks", "/Run"]]
    calls.clear()
    ok, msg = local_worker.restart(_worker(tmp_path, manager="task"), run=run, platform="win32",
                                   wait_gone=lambda pid, t: False, wait_new=lambda folder, old, t: 77)
    assert not ok and calls == [["schtasks", "/End"]] and "did not stop" in msg


def test_restart_refuses_busy_and_hand_started_workers(tmp_path):
    never = lambda cmd, **kw: pytest.fail("must not run anything")  # noqa: E731
    ok, msg = local_worker.restart(_worker(tmp_path, running=["big batch"]), run=never, platform="darwin")
    assert not ok and "big batch" in msg
    ok, msg = local_worker.restart(_worker(tmp_path, manager="manual"), run=never, platform="linux")
    assert not ok and "started by hand" in msg and "Ctrl+C" in msg


def test_wait_helpers(fake_worker):
    state, _repo, proc = fake_worker
    assert local_worker._wait_gone(proc.pid, 0.2) is False
    proc.kill()
    proc.wait(10)
    assert local_worker._wait_gone(proc.pid, 5) is True
    t0 = time.monotonic()
    assert local_worker._wait_new(state, proc.pid, 0.6) is None and time.monotonic() - t0 >= 0.5
