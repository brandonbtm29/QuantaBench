"""Dynamic CREST threads (dist.threads): the width policy, the thread budget's accounting, a small batch simulator
that compares the policy with fixed lanes, and parallel runs with a fake CREST that records the -T it was given."""

from __future__ import annotations

import heapq
import json
import random
import sys
import textwrap

import pytest

from quantabench import settings as settings_mod
from quantabench.chem import external, jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import lanes as L
from quantabench.dist import threads as T
from quantabench.dist.lanes import Lane


# ------------------------------------------------------------------------------------------ the policy
def test_one_molecule_takes_every_free_thread():
    assert T.choose_width(T.crest_cost(65), 4, 12, 12, rest=0.0) == 12
    assert T.choose_width(T.crest_cost(65), 2, 22, 22, rest=0.0) == 22


def test_a_dominant_molecule_takes_two_lanes_and_leaves_one():
    """The Mac's 3 x 4 lanes with 54950-05-9 (65 atoms) and 61119-62-8 (59) left: the large one takes 8 threads,
    the other lane keeps its 4 (all 12 for the first would leave the second nothing)."""
    big, other = T.crest_cost(65), T.crest_cost(59)
    assert T.choose_width(big, 4, 12, 12, rest=other * 4) == 8


def test_a_balanced_batch_keeps_the_lane_width():
    """20 molecules of similar size on 3 x 4 lanes: no molecule sets the end of the batch alone."""
    rest = sum(T.crest_cost(n) for n in range(40, 59)) * 4
    assert T.choose_width(T.crest_cost(60), 4, 12, 12, rest) == 4


def test_widths_are_whole_lanes_within_the_free_threads():
    for free in range(4, 14):
        w = T.choose_width(T.crest_cost(80), 4, free, 13, rest=0.0)
        assert w % 4 == 0 and 4 <= w <= free
    assert T.choose_width(T.crest_cost(80), 4, 7, 12, rest=0.0) == 4        # less than two lanes free
    assert T.choose_width(T.crest_cost(80), 3, 2, 12, rest=0.0) == 3        # never below the lane's own


def test_only_a_clearly_dominant_molecule_is_widened():
    """A molecule is widened only when its predicted time at the lane's width is more than DOMINANCE times the
    rest's work over the other threads (the size model scatters x1.8), and only to a width that predicts a clearly
    earlier end (MARGIN)."""
    base, budget, cost = 4, 12, 1.0
    rest = 1e-3
    seen = set()
    for _ in range(80):
        w = T.choose_width(cost, base, 12, budget, rest)
        seen.add(w)
        dominant = cost > T.DOMINANCE * rest / (budget - base)
        e4 = T.predicted_end(cost, 4, base, budget, rest)
        ew = T.predicted_end(cost, w, base, budget, rest)
        assert (w > base) == (dominant and T.predicted_end(cost, 8, base, budget, rest) < e4 * (1 - T.MARGIN))
        assert w == base or ew < e4 * (1 - T.MARGIN)
        rest *= 1.15
    assert seen == {4, 8}


# ------------------------------------------------------------------------------------- the accounting
def test_thread_budget_borrow_and_return():
    b = T.ThreadBudget(12)
    b.take("cpu1", 4, 4)
    assert b.in_use() == 8 and b.free() == 4 and b.borrowed("cpu1") == 4
    b.take("cpu2", 4)
    assert b.free() == 0
    with pytest.raises(ValueError, match="only 0 of 12 free"):
        b.take("cpu3", 4)                                   # never above the budget
    with pytest.raises(ValueError, match="already holds"):
        b.take("cpu2", 1)
    assert b.give_back("cpu1") == 4 and b.free() == 4 and b.give_back("cpu1") == 0
    b.take("cpu3", 4)
    assert b.release("cpu1") == 4 and b.release("cpu2") == 4 and b.release("cpu3") == 4
    assert b.in_use() == 0 and b.release("cpu3") == 0


def test_dynamic_threads_lifecycle_and_calibration():
    sizes = {"big": 65, "mid": 59, "small": 20}
    d = T.DynamicThreads(12, lambda m: T.crest_cost(sizes[m]))
    assert d.width(4, "big", ["mid"], now=0.0) == 8
    d.start("cpu1", 4, 8, "big", now=0.0)
    assert not d.can_start(8) and d.can_start(4)
    assert d.width(4, "mid", [], now=1.0) == 4                       # only one lane's worth left
    d.start("cpu2", 4, 4, "mid", now=1.0)
    assert not d.can_start(4)                                         # the third lane waits for its threads
    d.crest_started("cpu1", now=2.0)
    assert d.running_work(4, now=3.0) > 0
    back, seconds = d.crest_done("cpu1", now=102.0)
    assert (back, seconds) == (4, 100.0) and d.can_start(4) and d.samples
    assert d.crest_done("cpu1", now=103.0) == (0, 0.0)               # once
    assert d.finish("cpu1") == 4 and d.finish("cpu2") == 4 and d.pool.in_use() == 0
    assert d.widened == 1 and "1 CREST call(s) ran on borrowed threads" in d.summary()


def test_a_crashed_lane_returns_its_borrowed_threads():
    d = T.DynamicThreads(6, lambda m: 1e6)
    d.start("cpu1", 2, 6, "big", now=0.0)
    assert d.pool.free() == 0
    assert d.finish("cpu1", now=5.0) == 6                             # died during CREST: everything back
    assert d.pool.free() == 6 and d.borrowed_thread_s == pytest.approx(20.0)


@pytest.mark.parametrize("budget,base", [(12, 4), (13, 4), (22, 2), (8, 1), (3, 1), (24, 3)])
def test_random_runs_never_oversubscribe(budget, base):
    """Random sizes, finishes and crashes, the parent's call order: the threads in use never exceed the budget and
    all of them are back at the end."""
    rng = random.Random(budget * 31 + base)  # noqa: S311 - a test's reproducible draws
    sizes = {f"m{i}": rng.randint(5, 120) for i in range(60)}
    d = T.DynamicThreads(budget, lambda m: T.crest_cost(sizes[m]))
    lanes = [f"cpu{i}" for i in range(budget // base)]
    queue = list(sorted(sizes, key=lambda m: -sizes[m]))
    busy: dict[str, str] = {}
    in_crest: set[str] = set()
    t = 0.0
    while queue or busy:
        t += 1.0
        for lane in lanes:
            if lane not in busy and queue and d.can_start(base):
                m = queue.pop(0)
                w = d.width(base, m, queue, now=t)
                d.start(lane, base, w, m, now=t)
                busy[lane] = m
                in_crest.add(lane)
                assert d.pool.in_use() <= budget
        lane = rng.choice(sorted(busy))
        event = rng.random()
        if lane in in_crest and event < 0.5:
            d.crest_done(lane, now=t)
            in_crest.discard(lane)
        elif event < 0.9 or lane not in in_crest:
            d.finish(lane, now=t)                       # result, error, or a lane that died or timed out
            busy.pop(lane)
            in_crest.discard(lane)
        else:
            d.finish(lane, now=t)                       # crash during CREST: the molecule goes back to the queue
            queue.append(busy.pop(lane))
            in_crest.discard(lane)
        assert 0 <= d.pool.in_use() <= budget
    assert d.pool.in_use() == 0 and not d.pool.held


# ------------------------------------------------------------------- the policy in a simulated batch
def simulate(sizes, budget, base, dynamic, *, seconds_per_cost=1e-4, speedup=0.6, rest_fraction=0.1, noise=None):
    """Makespan of a batch: lanes of ``base`` threads within ``budget``, largest first, each molecule's CREST at the
    width the policy gives it (its true time follows ``speedup``), then ``rest_fraction`` of its base-width time on
    the lane's own threads. ``noise``: per-molecule factor of the true time (what the size model does not know)."""
    noise = noise or [1.0] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda j: -sizes[j])
    names = [f"m{j}" for j in range(len(sizes))]
    d = T.DynamicThreads(budget, lambda m: T.crest_cost(sizes[int(m[1:])]))
    lanes = [f"cpu{i}" for i in range(budget // base)]
    idle, events, queue, t = list(lanes), [], [names[j] for j in order], 0.0
    while True:
        while idle and queue and d.can_start(base):
            m, lane = queue.pop(0), idle.pop(0)
            w = d.width(base, m, queue, now=t) if dynamic else base
            d.start(lane, base, w, m, now=t)
            assert d.pool.in_use() <= budget
            j = int(m[1:])
            tb = seconds_per_cost * T.crest_cost(sizes[j]) * noise[j]
            heapq.heappush(events, (t + tb / (w / base) ** speedup, 0, lane, tb))
        if not events:
            return t
        t, kind, lane, tb = heapq.heappop(events)
        if kind == 0:
            d.crest_done(lane, now=t)
            heapq.heappush(events, (t + rest_fraction * tb, 1, lane, tb))
        else:
            d.finish(lane, now=t)
            idle.append(lane)


def test_simulated_narrow_lanes_with_a_dominant_molecule():
    """NOVA's 11 x 2 CREST plan and the 20-PFAS sizes (atoms with H): the 65-atom molecule on 2 threads sets the end
    of the batch; widened, the batch ends clearly earlier."""
    sizes = [49, 54, 56, 65, 50, 46, 37, 46, 39, 59, 8, 20, 18, 24, 25, 30, 30, 29, 31, 44]
    fixed = simulate(sizes, 22, 2, False)
    dyn = simulate(sizes, 22, 2, True)
    assert dyn < 0.8 * fixed


def test_simulated_few_molecules_get_the_idle_threads():
    sizes = [65, 56, 44, 25]
    assert simulate(sizes, 22, 2, True) < 0.6 * simulate(sizes, 22, 2, False)


def test_simulated_large_batch_is_unchanged():
    """Thousands of molecules with a long tail: every lane is busy until the queue is nearly empty, so the policy
    changes (almost) nothing, also when the size model is off by a factor of 1.8 per molecule."""
    rng = random.Random(3)  # noqa: S311 - a test's reproducible draws
    sizes = [min(130, max(5, int(rng.lognormvariate(3.5, 0.45)))) for _ in range(2000)]
    noise = [rng.lognormvariate(0.0, 0.6) for _ in sizes]
    for budget, base in ((12, 4), (22, 2)):
        fixed = simulate(sizes, budget, base, False, noise=noise)
        dyn = simulate(sizes, budget, base, True, noise=noise)
        assert dyn <= 1.01 * fixed, (budget, base, dyn / fixed)


def test_simulated_balanced_mac_batch_is_unchanged():
    sizes = [49, 54, 56, 65, 50, 46, 37, 46, 39, 59, 8, 20, 18, 24, 25, 30, 30, 29, 31, 44]
    assert simulate(sizes, 12, 4, True) <= 1.001 * simulate(sizes, 12, 4, False)


# -------------------------------------------------------------- parallel runs with a fake CREST
FAKE_CREST = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if "--version" in args:
    print("       Version 3.0.2, fake")
    sys.exit(0)
t0 = time.time()
time.sleep(float(os.environ.get("FAKE_CREST_SLEEP", "0")))
lines = Path(args[0]).read_text().splitlines()
n = int(lines[0])
Path("crest_conformers.xyz").write_text("\n".join([f"  {n}", "  -11.30"] + lines[2:2 + n]) + "\n")
with open(os.environ["FAKE_CREST_LOG"], "a") as f:
    f.write(json.dumps({"T": int(args[args.index("-T") + 1]), "omp": os.environ.get("OMP_NUM_THREADS"),
                        "atoms": n, "t0": t0, "t1": time.time()}) + "\n")
print(" CREST terminated normally.")
'''

PFHPA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"         # 23 atoms with H
PFBA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"                                  # 14
ETHANOL = "CCO"                                                        # 9


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
    return pipeline.BuildConfig(**{"method": "mmff", "conformer_search": "crest", "crest_method": "gfnff",
                                   "max_conformers": 2, "refine_top": 1, "xtb_properties": False, "threads": 3,
                                   "gpu": "off", "protonate": False, "output_dir": str(tmp_path / "runs"),
                                   "formats": ("xyz",), **kw})


def three_lanes():
    return [Lane(f"cpu{i}", "cpu", 1) for i in (1, 2, 3)]


def max_concurrent_threads(calls) -> int:
    edges = sorted([(c["t0"], c["T"]) for c in calls] + [(c["t1"], -c["T"]) for c in calls],
                   key=lambda e: (e[0], e[1]))
    now = peak = 0
    for _, dt in edges:
        now += dt
        peak = max(peak, now)
    return peak


def test_crest_borrows_idle_lane_threads(tmp_path, fake_crest, monkeypatch):
    """3 lanes x 1 thread, 2 molecules: the run uses 2 lanes, and the large molecule's CREST gets the third lane's
    thread (-T 2, OMP_NUM_THREADS 2); the threads in use never exceed 3."""
    monkeypatch.setenv("FAKE_CREST_SLEEP", "0.6")
    logs = []
    inputs = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA")]
    results, _ = jobs.run_resumable(inputs, crest_cfg(tmp_path), tmp_path / "run", log=logs.append,
                                    lanes=three_lanes())
    assert all(r.status == "ok" and r.conformer_search == "crest" for r in results), [r.error for r in results]
    calls = {c["atoms"]: c for c in fake_crest()}
    assert calls[23]["T"] == 2 and calls[23]["omp"] == "2"
    assert calls[9]["T"] in (1, 2)
    assert max_concurrent_threads(list(calls.values())) <= 3
    assert any("Dynamic threads: a CREST search may also use idle lanes' threads" in m and "3 threads in all" in m
               for m in logs)
    assert any("PFHpA: CREST with 2 threads (borrowed 1 from idle lanes)" in m for m in logs)
    assert any("PFHpA: conformer search done in" in m and "1 borrowed thread(s) back" in m for m in logs)
    assert any(m.startswith("Dynamic threads: ") and "ran on borrowed threads" in m for m in logs)


def test_dynamic_threads_off_keeps_the_lane_width(tmp_path, fake_crest):
    logs = []
    inputs = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA")]
    results, _ = jobs.run_resumable(inputs, crest_cfg(tmp_path), tmp_path / "run", log=logs.append,
                                    lanes=three_lanes(), dynamic_threads=False)
    assert all(r.status == "ok" for r in results)
    assert sorted(c["T"] for c in fake_crest()) == [1, 1]
    assert not any("Dynamic threads" in m or "borrowed" in m for m in logs)


def test_dynamic_threads_setting_off(tmp_path, fake_crest):
    s = settings_mod.load()
    s.resources.dynamic_threads = False
    settings_mod.save(s)
    assert L.dynamic_threads_for(crest_cfg(tmp_path), three_lanes()) == 0
    assert L.dynamic_threads_for(crest_cfg(tmp_path), three_lanes(), enabled=True) == 3


def test_dynamic_threads_only_for_crest_on_cpu_lanes(tmp_path):
    assert L.dynamic_threads_for(crest_cfg(tmp_path), three_lanes(), enabled=True) == 3
    assert L.dynamic_threads_for(crest_cfg(tmp_path, conformer_search="rdkit"), three_lanes(), enabled=True) == 0
    assert L.dynamic_threads_for(crest_cfg(tmp_path), [Lane("gpu0", "cuda:0", 4)], enabled=True) == 0
    assert L.dynamic_threads_for(crest_cfg(tmp_path), [Lane("cpu1", "cpu", 4)], enabled=True) == 0   # no room
    assert L.dynamic_threads_for(crest_cfg(tmp_path), [Lane("cpu1", "cpu", 4)], enabled=True, budget=12) == 12


def test_one_molecule_gets_every_lane_of_the_plan(tmp_path, fake_crest):
    """A run of one molecule on a plan of 3 lanes runs in this process on one lane's threads; its CREST search gets
    all three."""
    logs = []
    results, _ = jobs.run_resumable([from_smiles(PFHPA, id="PFHpA")], crest_cfg(tmp_path), tmp_path / "run",
                                    log=logs.append, lanes=three_lanes())
    assert results[0].status == "ok"
    assert [c["T"] for c in fake_crest()] == [3]
    assert any("Dynamic threads: PFHpA: CREST with 3 threads (borrowed 2" in m for m in logs)


def test_a_lane_that_dies_returns_its_threads(tmp_path, fake_crest, monkeypatch):
    """The large molecule crashes its lane (twice: then it is marked failed) after it was given 2 threads; they come
    back with the lane's death, so the other molecules still run and the run ends."""
    monkeypatch.setenv(L.TEST_CRASH_ENV, "PFHpA")
    logs = []
    inputs = [from_smiles(ETHANOL, id="ethanol"), from_smiles(PFHPA, id="PFHpA"), from_smiles(PFBA, id="PFBA")]
    results, _ = jobs.run_resumable(inputs, crest_cfg(tmp_path), tmp_path / "run", log=logs.append,
                                    lanes=three_lanes())
    by_id = {r.input.id: r for r in results}
    assert by_id["PFHpA"].status == "failed" and "crashed the process" in by_id["PFHpA"].error
    assert by_id["ethanol"].status == "ok" and by_id["PFBA"].status == "ok"
    assert any("PFHpA: CREST with" in m and "borrowed" in m for m in logs)
    assert max_concurrent_threads(fake_crest()) <= 3


def test_lane_task_without_crest_threads_is_still_read(tmp_path):
    """The parent sends (k, id, text, crest_threads); 0 means the lane's own threads (no copy of the config)."""
    assert pipeline.BuildConfig().crest_threads == 0
    assert "crest_threads" not in jobs.SCIENCE_FIELDS


def test_lanes_cli_mentions_dynamic_threads(capsys, monkeypatch):
    from quantabench import resources

    s = settings_mod.Settings()
    s.build.conformer_search = "crest"
    settings_mod.save(s)
    monkeypatch.setattr(resources, "detect", lambda: resources.HardwareReport(
        os="test", python="3.12", cpu_name="test", physical_cores=4, logical_cores=4, ram_gb=16.0, wsl=False,
        torch_version="", torch_backend="cpu", gpus=[]))
    monkeypatch.setattr(resources, "logical_cores", lambda: 4)

    class Args:
        spec, method, threads, gpu, json = "2x2", None, 4, None, False

    assert L._cli(Args()) == 0
    assert "dynamic threads" in capsys.readouterr().out
