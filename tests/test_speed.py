"""Speed-aware lanes (dist.speed): the hold-back decision, the estimates, and the whole policy under a
simulated clock. No processes here; the real pool is tested in test_lanes.py."""
# ruff: noqa: S311 - seeded pseudo-random molecule sizes and timings for the simulation

import random
from collections import deque

import pytest

from quantabench.dist import profile as profile_mod
from quantabench.dist import speed
from quantabench.dist.lanes import Lane
from quantabench.dist.speed import LaneView, should_dispatch


def fast(name="gpu0", per_unit=1.0, **kw):
    return LaneView(name, per_unit, **kw)


def slow(name="cpu1", per_unit=10.0, **kw):
    return LaneView(name, per_unit, **kw)


def dispatch(lane, others, q, size=1.0):
    return should_dispatch(lane, others, [size] * q)


# ------------------------------------------------------------------------------------ the decision
def test_one_fast_four_slow_idle_fast_lane():
    """A lane 10x slower leaves the queue to one idle fast lane while it is shorter than 10 (slow / fast)."""
    others = [fast()] + [slow(f"cpu{i}") for i in (2, 3, 4)]
    me = slow()
    assert [dispatch(me, others, q) for q in (1, 2, 5, 8, 9)] == [False] * 5
    assert [dispatch(me, others, q) for q in (10, 11, 20, 500)] == [True] * 4
    # the other slow lanes are never counted on, busy or idle
    assert dispatch(me, [fast()], 9) is False and dispatch(me, [fast()], 10) is True
    assert dispatch(me, [fast(), slow("cpu2", busy_size=1.0, elapsed=9.9)], 9) is False


def test_one_fast_lane_busy():
    """A busy fast lane is free after its molecule's expected remainder: 3 s left, then 1 s per molecule,
    gives slots 3+1, ..., 3+q against 10 s: hold back for q <= 6."""
    me = slow()
    gpu = fast(busy_size=4.0, elapsed=1.0)              # 4 s molecule, 1 s in: 3 s left
    assert speed.residual(gpu) == pytest.approx(3.0)
    assert dispatch(me, [gpu], 6) is False and dispatch(me, [gpu], 7) is True


def test_two_fast_two_slow():
    me = slow("cpu1")
    others = [fast("gpu0"), fast("gpu1"), slow("cpu2")]
    # two fast lanes: slots 1,1,2,2,...,9,9 before 10 s
    assert dispatch(me, others, 18) is False and dispatch(me, others, 19) is True
    # one of them busy for 4 more seconds: 1..9 on gpu0 and 5..9 on gpu1, 14 in time
    others = [fast("gpu0"), fast("gpu1", busy_size=5.0, elapsed=1.0), slow("cpu2")]
    assert dispatch(me, others, 14) is False and dispatch(me, others, 15) is True
    # fast lanes of different speeds: 1 s and 2 s per molecule finish 9 + 4 in time
    others = [fast("gpu0", 1.0), fast("gpu1", 2.0), slow("cpu2")]
    assert dispatch(me, others, 13) is False and dispatch(me, others, 14) is True


def test_equal_speeds_never_hold_back():
    for others in ([slow("cpu2")], [slow("cpu2", busy_size=1.0, elapsed=9.99)], [slow("cpu2", 8.0)],
                   [slow(f"cpu{i}", busy_size=1.0, elapsed=5.0) for i in range(2, 8)]):
        assert all(dispatch(slow(), others, q) for q in (1, 2, 3, 50))
    # lanes within MIN_SPEEDUP (1.5x) of each other do not count as faster
    assert dispatch(slow(per_unit=10.0), [slow("cpu2", 10.0 / 1.49)], 1) is True
    assert dispatch(slow(per_unit=10.0), [slow("cpu2", 10.0 / 1.51)], 1) is False


def test_unknown_speeds_mean_todays_behavior():
    assert dispatch(LaneView("cpu1", None), [fast()], 1) is True                  # own speed unknown
    assert dispatch(slow(), [LaneView("gpu0", None)], 1) is True                  # the other's unknown
    assert dispatch(slow(), [], 1) is True
    assert should_dispatch(slow(), [fast()], []) is False                         # nothing to take


def test_unavailable_or_stalled_fast_lane_is_not_counted_on():
    me = slow()
    assert dispatch(me, [fast(available=False)], 1) is True                       # died, restarting, retired
    # a fast lane overdue on its molecule is assumed to need as long again as it is late: a 1 s molecule
    # running 5 s is free in 4 s (slot 5 < 10: hold), at 10.5 s in 9.5 s (slot 10.5: take it)
    assert dispatch(me, [fast(busy_size=1.0, elapsed=5.0)], 1) is False
    assert dispatch(me, [fast(busy_size=1.0, elapsed=10.5)], 1) is True


def test_molecule_sizes():
    """Times scale with size: the decision uses the size of each queued molecule."""
    me = slow(per_unit=1.0)                      # 1 s per atom
    gpu = fast(per_unit=0.1)                     # 0.1 s per atom
    # head of 20 atoms: 20 s on the slow lane; the GPU does 20 + 20 + 100 atoms in 14 s: hold back
    assert should_dispatch(me, [gpu], [20, 20, 100]) is False
    # a big molecule behind it takes the GPU past 20 s: help
    assert should_dispatch(me, [gpu], [20, 20, 170]) is True
    # a small head molecule: the slow lane would finish it in 3 s, the GPU cannot do all three by then
    assert should_dispatch(me, [gpu], [3, 20, 20]) is True


def test_long_queue_is_cheap():
    """A long queue stops the simulation once the fast lanes run out of time (the sizes are read lazily)."""
    read = []

    def sizes():
        for _ in range(1_000_000):
            read.append(1)
            yield 1.0

    assert should_dispatch(slow(), [fast()], sizes()) is True and len(read) <= 11


# ------------------------------------------------------------------------------------- estimates
def lanes_nova():
    return [Lane("gpu0", "cuda:0", 4)] + [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 5)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_policy_estimates_measured_seeded_unknown():
    clock = Clock()
    pol = speed.SpeedPolicy(lanes_nova(), {"gpu0": 0.01, "cpu1": 0.1}, clock=clock)
    assert pol.source("gpu0") == "benchmark" and pol.source("cpu3") == "benchmark"   # one seed per class
    assert pol.estimate("cpu3") == pytest.approx(0.1)
    # before anything finished: the smallest factor under which no running molecule is overdue
    pol.started("gpu0", 10.0)
    clock.t = 0.5
    assert pol.scale() == pytest.approx(5.0)                 # 0.5 s for a 0.1 s (benchmark) molecule
    assert pol.estimate("cpu1") == pytest.approx(0.5)
    # the first molecule (model loading) counts only until the class has later ones
    pol.finished("gpu0", ok=True)                           # 0.05 s per atom
    assert pol.measured(("cuda:0", 4)) == pytest.approx(0.05) and pol.scale() == pytest.approx(5.0)
    pol.started("gpu0", 10.0)
    clock.t = 0.7
    pol.finished("gpu0", ok=True)                           # 0.02 s per atom
    assert pol.estimate("gpu0") == pytest.approx(0.02)
    assert pol.estimate("cpu2") == pytest.approx(0.2)        # benchmark x (measured / benchmark of the GPU)
    # failures and lost molecules are not timed
    pol.started("cpu1", 10.0)
    clock.t = 0.8
    pol.finished("cpu1", ok=False)
    pol.started("cpu2", 10.0)
    pol.lost("cpu2")
    assert pol.source("cpu1") == "benchmark" and not pol.busy
    # lanes of one class share their measurements
    pol.started("cpu1", 10.0)
    clock.t = 10.8
    pol.finished("cpu1")
    assert pol.estimate("cpu4") == pytest.approx(1.0) and pol.source("cpu4") == "measured"
    assert "gpu0 0.40 s (measured)" in pol.summary(20.0) and "cpu1-cpu4 20 s (measured)" in pol.summary(20.0)


def test_policy_without_seeds_holds_back_once_measured():
    clock = Clock()
    pol = speed.SpeedPolicy(lanes_nova(), None, clock=clock)
    states = {x.name: speed.READY for x in lanes_nova()}
    assert pol.allow("cpu1", [20.0] * 3, states)            # unknown: take it
    for name, dt in (("gpu0", 1.0), ("cpu1", 10.0)):
        pol.started(name, 20.0)
        clock.t += dt
        pol.finished(name)
    assert not pol.allow("cpu1", [20.0] * 3, states)        # the GPU does 3 in 3 s, cpu1 one in 10 s
    msg = pol.hold_message(["cpu1"], 3, 20.0, states)
    assert "cpu1 (~10 s per molecule, measured)" in msg and "gpu0 (~1.0 s per molecule, measured)" in msg
    assert "--no-speed-aware-lanes" in msg
    # the GPU lane gone: take it; still starting for the first time: counted on
    assert pol.allow("cpu1", [20.0] * 3, {**states, "gpu0": speed.DOWN})
    assert not pol.allow("cpu1", [20.0] * 3, {**states, "gpu0": speed.STARTING})
    # one kind of lane never holds back for its own kind
    assert pol.allow("cpu1", [20.0], {n: (speed.READY if n.startswith("cpu") else speed.DOWN) for n in states})


def test_registry_seeds(tmp_path):
    p = profile_mod.MachineProfile(name="btm-nova", throughput={"aimnet2@cuda:0": 7200.0, "aimnet2@cpu": 720.0,
                                                                "mmff@cpu": 36000.0})
    lanes = lanes_nova()
    seeds = speed.registry_seeds(lanes, "aimnet2", profile=p)
    unit = speed.bench_size()
    assert 10 < unit < 30                                       # PFAS anions, atoms with H
    assert seeds["gpu0"] == pytest.approx(0.5 / unit) and seeds["cpu1"] == pytest.approx(5.0 / unit)
    assert speed.registry_seeds(lanes, "gfn1-xtb", profile=p) == {x.name: None for x in lanes}
    assert speed.registry_seeds(lanes, "aimnet2") == {x.name: None for x in lanes}     # empty registry here
    profile_mod.save_profile(profile_mod.MachineProfile(name=profile_mod.machine_name(),
                                                        throughput={"aimnet2@cpu": 720.0}))
    assert speed.registry_seeds(lanes, "aimnet2")["cpu2"] == pytest.approx(5.0 / unit)


def test_molecule_size():
    from quantabench.chem.inputs import MoleculeInput, from_smiles

    assert speed.molecule_size(from_smiles("CCO", id="e")) == 9.0
    assert speed.molecule_size(from_smiles("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", id="pfba")) == 14.0     # C4HF7O2
    assert speed.molecule_size(MoleculeInput(id="x", smiles="not a smiles")) == speed.DEFAULT_SIZE


# ------------------------------------------------------------------------ the policy, simulated clock
def simulate(n, lanes, per_atom, sizes, *, policy, seeds=None, startup=None, noise=0.0, rng_seed=0, poll=0.5,
             down_at=None, stall=None):
    """``run_parallel``'s dispatch loop under a simulated clock: each free, ready lane (in list order) asks the
    policy and takes the head of the queue; the loop wakes when a lane finishes or becomes ready, and every
    ``poll`` seconds while molecules are queued (``lanes.POLL_S``). ``per_atom``: each lane's true seconds per
    atom, ``noise``: log-normal scatter per molecule and device. ``down_at=(lane, t)``: that lane dies at t
    (its molecule goes back to the queue). ``stall=(lane, k, extra)``: that lane's k-th molecule takes
    ``extra`` seconds longer. Returns (makespan, molecules per lane, start times per lane)."""
    startup = startup or {}
    now = [0.0]
    pol = speed.SpeedPolicy(lanes, seeds, clock=lambda: now[0]) if policy else None
    devices = {x.device for x in lanes}         # the same scatter per molecule and device whatever the lanes
    fac = {(m, d): random.Random(f"{rng_seed}:{m}:{d}").lognormvariate(0.0, noise) if noise else 1.0
           for m in range(n) for d in devices}
    queue = deque(range(n))
    busy: dict[str, tuple[int, float]] = {}
    ready_at = {x.name: startup.get(x.name, 0.0) for x in lanes}
    down: set[str] = set()
    ends: list[float] = []
    count = {x.name: 0 for x in lanes}
    starts: dict[str, list[float]] = {x.name: [] for x in lanes}
    while queue or busy:
        t = now[0]
        if down_at and t >= down_at[1] and down_at[0] not in down:
            down.add(down_at[0])
            if down_at[0] in busy:
                m, _ = busy.pop(down_at[0])
                queue.append(m)
                if pol:
                    pol.lost(down_at[0])
        states = {x.name: speed.DOWN if x.name in down else speed.READY if t >= ready_at[x.name]
                  else speed.STARTING for x in lanes}
        for x in lanes:
            if queue and states[x.name] == speed.READY and x.name not in busy:
                if pol and not pol.allow(x.name, (sizes[m] for m in queue), states):
                    continue
                m = queue.popleft()
                dur = per_atom[x.name] * sizes[m] * fac[(m, x.device)]
                if stall and stall[0] == x.name and stall[1] == count[x.name]:
                    dur += stall[2]
                busy[x.name] = (m, t + dur)
                count[x.name] += 1
                starts[x.name].append(t)
                if pol:
                    pol.started(x.name, sizes[m])
        if not busy and not queue:
            break
        wake = [e for _, e in busy.values()] + [r for r in ready_at.values() if r > t]
        if queue:
            wake.append(t + poll)
        if down_at and down_at[1] > t:
            wake.append(down_at[1])
        now[0] = min(wake)
        for name, (_m, end) in list(busy.items()):
            if end <= now[0] + 1e-9:
                del busy[name]
                ends.append(end)
                if pol:
                    pol.finished(name, ok=True)
    return max(ends), count, starts


GPU_S_PER_ATOM = 0.25        # ~5 s per 20-atom PFAS on the GPU lane, 50 s on a CPU lane
NOVA_STARTUP = {"gpu0": 6.0, **{f"cpu{i}": 4.0 for i in range(1, 5)}}


def nova_run(n, *, policy, seeds=True, bench_ratio=10.0, rng_seed=0, noise=0.1, gpu_only=False, uniform=False,
             **kw):
    """1 GPU lane + 4 CPU lanes, the GPU 10x faster per atom; molecules of 10-40 atoms (C2-C10 PFAS-like) or
    all of 20; lanes ready after 4-6 s. The benchmark seeds are 5x faster than the build (a shorter protocol)
    with ``bench_ratio`` between the devices."""
    rng = random.Random(1000 + rng_seed)
    sizes = [20] * n if uniform else [rng.randint(10, 40) for _ in range(n)]
    lanes = lanes_nova()[:1] if gpu_only else lanes_nova()
    per_atom = {"gpu0": GPU_S_PER_ATOM, **{f"cpu{i}": 10 * GPU_S_PER_ATOM for i in range(1, 5)}}
    bench = {"gpu0": GPU_S_PER_ATOM / 5, **{f"cpu{i}": bench_ratio * GPU_S_PER_ATOM / 5 for i in range(1, 5)}}
    return simulate(n, lanes, per_atom, sizes, policy=policy, seeds=bench if seeds else None,
                    startup=NOVA_STARTUP, noise=noise, rng_seed=rng_seed, **kw)


@pytest.mark.parametrize("n", [8, 50, 500])
def test_simulated_makespans_gpu_ten_times_faster(n):
    """GPU 10x faster, 10 % scatter per molecule, speeds seeded from the benchmark: with the policy the batch
    never ends later than without it, nor later than on the GPU lane alone (5 sets of molecule sizes)."""
    for rng_seed in range(5):
        on, count, _ = nova_run(n, policy=True, rng_seed=rng_seed)
        off, _, _ = nova_run(n, policy=False, rng_seed=rng_seed)
        alone, _, _ = nova_run(n, policy=False, rng_seed=rng_seed, gpu_only=True)
        assert on <= off + 1e-6 and on <= alone + 1e-6, (n, rng_seed, on, off, alone, count)
        if n == 8:
            assert count["gpu0"] >= 7
        else:
            assert min(count[f"cpu{i}"] for i in range(1, 5)) >= 1          # the CPU lanes help in the middle


def test_simulated_makespans_report():
    """The numbers behind the backlog entry (20-atom molecules, exact times, benchmark-seeded)."""
    rows = {}
    for n in (8, 50, 500):
        rows[n] = tuple(round(nova_run(n, policy=p, noise=0.0, uniform=True, gpu_only=g)[0], 1)
                        for p, g in ((True, False), (False, False), (False, True)))
    print("\nmakespan in s (policy, no policy, GPU lane alone):", rows)
    for on, off, alone in rows.values():
        assert on <= off and on <= alone
    assert rows[8][0] == rows[8][2] < rows[8][1] and rows[50][0] < rows[50][1]


@pytest.mark.parametrize("n", [8, 50, 500])
def test_simulated_estimate_errors_cost_little(n):
    """A benchmark ratio 10 % off (9x for 10x) and 20 % scatter: the policy can lose a little on one run
    (it trusts the estimates) but not on average, and never much."""
    ratios = []
    for rng_seed in range(10):
        on, _, _ = nova_run(n, policy=True, bench_ratio=9.0, noise=0.2, rng_seed=rng_seed)
        off, _, _ = nova_run(n, policy=False, noise=0.2, rng_seed=rng_seed)
        alone, _, _ = nova_run(n, policy=False, noise=0.2, rng_seed=rng_seed, gpu_only=True)
        ratios.append((on / off, on / min(off, alone)))
    assert sum(r for r, _ in ratios) / len(ratios) <= 1.0
    assert max(r for _, r in ratios) <= 1.3, ratios


@pytest.mark.parametrize("n", [8, 50, 500])
def test_simulated_without_benchmark_never_slower(n):
    """No benchmark in the registry: speeds are unknown until measured, so the first molecules go out as
    today; from then on the policy only helps."""
    for rng_seed in range(3):
        on, _, _ = nova_run(n, policy=True, seeds=False, rng_seed=rng_seed)
        off, _, _ = nova_run(n, policy=False, seeds=False, rng_seed=rng_seed)
        assert on <= off + 1e-6, (n, rng_seed, on, off)


def test_simulated_equal_lanes_are_unchanged():
    lanes = [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]
    per_atom = {x.name: 1.0 for x in lanes}
    sizes = [random.Random(k).randint(10, 40) for k in range(40)]
    on = simulate(40, lanes, per_atom, sizes, policy=True, noise=0.3)
    off = simulate(40, lanes, per_atom, sizes, policy=False, noise=0.3)
    assert on == off


def test_simulated_fast_lane_dies_slow_lanes_finish():
    """The GPU lane dies at 20 s with molecules held back for it: the CPU lanes take them at once and the
    run ends."""
    makespan, count, starts = nova_run(8, policy=True, noise=0.0, uniform=True, down_at=("gpu0", 20.0))
    cpu_starts = sorted(t for name, ts in starts.items() if name != "gpu0" for t in ts)
    assert cpu_starts[0] == pytest.approx(20.0)                  # nothing before, at once after
    assert sum(count.values()) == 8 + 1 and count["gpu0"] == 3   # its third molecule went back to the queue
    assert makespan < 20.0 + 2 * 10 * GPU_S_PER_ATOM * 20 + 1e-6


def test_simulated_fast_lane_stalls_slow_lanes_take_over():
    """The GPU lane's 2nd molecule hangs for 1000 s: the CPU lanes take the held molecules after about one
    CPU molecule's time, not after the hang."""
    makespan, count, starts = nova_run(8, policy=True, noise=0.0, uniform=True, stall=("gpu0", 1, 1000.0))
    cpu_starts = sorted(t for name, ts in starts.items() if name != "gpu0" for t in ts)
    t_cpu = 10 * GPU_S_PER_ATOM * 20
    assert cpu_starts and cpu_starts[0] < 6.0 + 5.0 + t_cpu + 1.0
    assert count["gpu0"] == 2 and sum(count.values()) == 8
    assert max(t for ts in starts.values() for t in ts) < 1000.0     # everything else started during the hang


def test_cli_build_flag():
    from quantabench.cli import build_parser

    p = build_parser()
    assert p.parse_args(["build", "CCO"]).speed_aware_lanes is None                 # from settings
    assert p.parse_args(["build", "CCO", "--no-speed-aware-lanes"]).speed_aware_lanes is False
    assert p.parse_args(["build", "CCO", "--speed-aware-lanes"]).speed_aware_lanes is True


def test_registry_seeds_in_water_and_with_the_mixed_set():
    p = profile_mod.MachineProfile(name="btm-nova", throughput={"tblite-gfn2@cpu": 7200.0,
                                                                "tblite-gfn2+water@cpu": 720.0},
                                   benchmarks={"tblite-gfn2+water@cpu": {"set": "mixed-10"}})
    lanes = lanes_nova()
    pfas, mixed = speed.bench_size("pfas"), speed.bench_size("mixed")
    assert mixed > pfas and speed.bench_size() == pfas
    gas = speed.registry_seeds(lanes, "tblite-gfn2", profile=p)
    water = speed.registry_seeds(lanes, "tblite-gfn2", profile=p, solvent="water")
    assert gas["cpu1"] == pytest.approx(0.5 / pfas) and water["cpu1"] == pytest.approx(5.0 / mixed)
    assert gas["gpu0"] is None and water["gpu0"] is None
    # AIMNet2 has no water model: a run in water uses (and only uses) its gas-phase keys
    q = profile_mod.MachineProfile(name="btm-nova", throughput={"aimnet2@cpu": 720.0, "aimnet2+water@cpu": 1.0})
    assert speed.registry_seeds(lanes, "aimnet2", profile=q, solvent="water")["cpu1"] == pytest.approx(5.0 / pfas)
