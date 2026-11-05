"""Weighted splitting (dist.plan) and weighted chunks in resumable runs (chem.jobs)."""

import json

import pytest

from quantabench import cli
from quantabench.chem import export, jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import plan
from quantabench.dist.profile import MachineProfile, machine_name, save_profile

IDS = [f"DTXSID{n:07d}" for n in range(20000)]
MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F"),
        ("methanol", "CO"), ("propanol", "CCCO"), ("TFA", "OC(=O)C(F)(F)F"), ("water", "O")]


def prof(name, cores=8, engines=None, **throughput):
    return MachineProfile(name=name, physical_cores=cores, logical_cores=2 * cores,
                          engines=engines or {"mmff": True, "gfn1-xtb": True},
                          throughput={k.replace("_at_", "@").replace("_", "-"): v for k, v in throughput.items()},
                          updated="2026-11-04T00:00:00+00:00")


# ------------------------------------------------------------------------------------ parsing
def test_positional_and_named_weights():
    w = plan.parse_weights("3,1")
    assert dict(w) == {"1": 3.0, "2": 1.0} and not w.named and w.spec() == "3,1"
    w = plan.parse_weights(" BTM NOVA=3, brandons-macbook-pro=1.5 ")
    assert dict(w) == {"btm-nova": 3.0, "brandons-macbook-pro": 1.5} and w.named
    assert list(w) == ["brandons-macbook-pro", "btm-nova"]               # a mapping: chunks in name order
    assert w.spec() == "brandons-macbook-pro=1.5,btm-nova=3"
    for bad in ("", "3,x", "3,-1", "0,0", "a=3,1", "a=1,a=2", "3,auto", "=2", "inf,1"):
        with pytest.raises(ValueError):
            plan.parse_weights(bad, "mmff", [])


def test_auto_weights_from_benchmarks():
    profiles = [prof("brandons-macbook-pro", mmff_at_cpu=1000.0, gfn1_xtb_at_cpu=200.0),
                prof("btm-nova", mmff_at_cpu=3000.0, gfn1_xtb_at_cpu=300.0, **{"gfn1_xtb_at_cuda:0": 900.0})]
    w = plan.parse_weights("auto", "mmff", profiles)
    assert list(w) == ["brandons-macbook-pro", "btm-nova"]          # sorted by name on every machine
    assert dict(w) == {"brandons-macbook-pro": 1000.0, "btm-nova": 3000.0} and not w.notes
    w = plan.parse_weights("auto", "gfn1-xtb", profiles)
    assert w["btm-nova"] == 900.0 and w.sources["btm-nova"] == "benchmark gfn1-xtb@cuda:0"   # fastest device
    # measured numbers are rounded to 3 digits, so the printed explicit spec reproduces the split exactly
    profiles[0].throughput["mmff@cpu"] = 1234.567
    w = plan.parse_weights("auto", "mmff", profiles)
    assert w["brandons-macbook-pro"] == 1230.0 and plan.parse_weights(w.spec()) == w
    # named entries may say auto, mixed with numbers
    w = plan.parse_weights("btm-nova=auto,brandons-macbook-pro=500", "mmff", profiles)
    assert dict(w) == {"btm-nova": 3000.0, "brandons-macbook-pro": 500.0}
    with pytest.raises(ValueError, match="explicit --method"):
        plan.parse_weights("auto", "auto", profiles)
    with pytest.raises(ValueError, match="not in the machine registry"):
        plan.parse_weights("nobody=auto", "mmff", profiles)
    with pytest.raises(ValueError, match="registry is empty"):
        plan.parse_weights("auto", "mmff", [])
    # a WSL profile shares its computer's cores: plain auto leaves it out (and says so), naming it includes it
    wsl = prof("btm-nova-wsl", mmff_at_cpu=2000.0)
    wsl.wsl = True
    old = prof("old-laptop", mmff_at_cpu=10.0)
    old.updated = "2025-01-01T00:00:00+00:00"
    w = plan.parse_weights("auto", "mmff", [*profiles, wsl, old])
    assert "btm-nova-wsl" not in w and any("left out WSL" in n for n in w.notes)
    assert any("old-laptop's profile is" in n for n in w.notes)
    assert plan.parse_weights("btm-nova-wsl=auto,btm-nova=1", "mmff", [wsl])["btm-nova-wsl"] == 2000.0


def test_auto_weights_without_benchmarks_are_estimated_and_say_so():
    profiles = [prof("a", cores=12, mmff_at_cpu=5000.0), prof("b", cores=8)]
    w = plan.parse_weights("auto", "mmff", profiles)
    assert w["a"] == 5000.0 and w["b"] == plan.round_sig(8 * plan.ESTIMATE_PER_CORE["mmff"])
    assert w.sources["b"].startswith("estimate: 8 cores") and "no mmff benchmark for b" in w.notes[0]
    w = plan.parse_weights("auto", "gfn1-xtb", profiles)                  # nobody measured gfn1-xtb
    assert w["a"] / w["b"] == pytest.approx(12 / 8) and "no machine has a gfn1-xtb benchmark" in w.notes[0]
    # a machine that cannot run the method gets weight 0 (and the note says so)
    profiles.append(prof("c", engines={"mmff": True, "gfn1-xtb": False}, gfn1_xtb_at_cpu=999.0))
    w = plan.parse_weights("auto", "gfn1-xtb", profiles)
    assert w["c"] == 0.0 and any("c cannot run gfn1-xtb" in n for n in w.notes)
    text = plan.explain(w, 1000)
    assert "estimate: 12 cores" in text and "note:" in text


# ------------------------------------------------------------------------------------ splitting
def test_weighted_split_converges_and_is_stable():
    parts = plan.split(IDS, {"nova": 3.0, "mac": 1.0})
    assert len(parts["nova"]) + len(parts["mac"]) == len(IDS)
    assert len(parts["nova"]) / len(IDS) == pytest.approx(0.75, abs=0.01)
    small = plan.split(IDS[:500], {"nova": 3.0, "mac": 1.0})
    assert small["nova"] == [i for i in parts["nova"] if i in set(IDS[:500])]      # growing the list moves nothing
    assert plan.split(list(reversed(IDS[:500])), {"nova": 3.0, "mac": 1.0})["mac"] == list(reversed(small["mac"]))
    for weights in ([1, 1, 1], [5, 2, 1], [0.1, 0.7]):
        counts = [0] * len(weights)
        for i in IDS:
            counts[plan.weighted_index(i, weights)] += 1
        for c, w in zip(counts, weights, strict=True):
            assert c / len(IDS) == pytest.approx(w / sum(weights), abs=0.012)


def test_zero_weight_target_changes_nothing():
    base = plan.split(IDS[:3000], {"nova": 3.0, "mac": 1.0})
    for extra in ({"nova": 3.0, "mac": 1.0, "wsl": 0.0}, {"nova": 3.0, "wsl": 0.0, "mac": 1.0},
                  {"wsl": 0.0, "nova": 3.0, "mac": 1.0}):
        got = plan.split(IDS[:3000], extra)
        assert got["wsl"] == [] and got["nova"] == base["nova"] and got["mac"] == base["mac"]
    with pytest.raises(ValueError):
        plan.weighted_index("x", [0, 0])


def test_explain_shares_and_exact_counts():
    w = plan.parse_weights("btm-nova=3,brandons-macbook-pro=1")
    text = plan.explain(w, 4700, this="btm-nova")
    assert "75.0%" in text and "~3,525" in text and "<- this machine" in text
    exact = plan.explain(w, IDS[:4700])
    counts = [int(line.split()[4].replace(",", "")) for line in exact.splitlines()[1:]]
    assert sum(counts) == 4700 and "~" not in exact
    assert "Split by weight:" in plan.explain(w, None)


def test_resolve_chunk():
    profiles = [prof("btm-nova", mmff_at_cpu=3000.0), prof("brandons-macbook-pro", mmff_at_cpu=1000.0)]
    chunk, w, warn = plan.resolve_chunk("3,1", (0, 2), "mmff", profiles, this="btm-nova")
    assert chunk == (0, 2, (3.0, 1.0)) and not warn
    chunk, w, warn = plan.resolve_chunk("auto", None, "mmff", profiles, this="BTM-NOVA")   # finds its own share
    assert chunk == (1, 2, (1000.0, 3000.0)) and not warn
    chunk, w, warn = plan.resolve_chunk("auto", (0, 2), "mmff", profiles, this="btm-nova")
    assert chunk[0] == 0 and "brandons-macbook-pro's share" in warn[0]
    with pytest.raises(ValueError, match="need --chunk"):
        plan.resolve_chunk("3,1", None, "mmff", profiles)
    with pytest.raises(ValueError, match="does not match"):
        plan.resolve_chunk("3,1", (0, 3), "mmff", profiles)
    with pytest.raises(ValueError, match="not in --chunk-weights"):
        plan.resolve_chunk("a=1,b=2", None, "mmff", profiles, this="btm-nova")
    with pytest.raises(ValueError, match="weight 0"):
        plan.resolve_chunk("0,1", (0, 2), "mmff", profiles)
    text = plan.chunk_plan(w, chunk, IDS[:4700], warn, this="btm-nova")
    mine = sum(1 for i in IDS[:4700] if plan.weighted_index(i, chunk[2]) == 0)
    assert f"this machine (btm-nova, chunk 1/2, weight 1000): {mine:,} of 4,700 molecules" in text
    assert "--chunk-weights brandons-macbook-pro=1000,btm-nova=3000" in text and "warning:" in text


def test_named_weights_in_any_order_give_one_split():
    """Two machines given the same named weights typed in different orders run complementary shares (in
    typed order they overlapped by a third and left a quarter of the list out)."""
    a, wa, _ = plan.resolve_chunk("btm-nova=3,brandons-macbook-pro=1", None, this="btm-nova")
    b, wb, _ = plan.resolve_chunk("brandons-macbook-pro=1,btm-nova=3", None, this="brandons-macbook-pro")
    assert list(wa) == list(wb) == ["brandons-macbook-pro", "btm-nova"]
    assert a == (1, 2, (1.0, 3.0)) and b == (0, 2, (1.0, 3.0))
    ids = IDS[:4000]
    mine_a = {i for i in ids if jobs.in_chunk(i, jobs.check_chunk(a))}
    mine_b = {i for i in ids if jobs.in_chunk(i, jobs.check_chunk(b))}
    assert not mine_a & mine_b and mine_a | mine_b == set(ids)
    # an explicit --chunk K/N counts in the same name order, and the plan says so
    c, w, warn = plan.resolve_chunk("btm-nova=3,brandons-macbook-pro=1", (1, 2), this="btm-nova")
    assert c == a and not warn
    text = plan.chunk_plan(w, c, ids, warn, this="btm-nova")
    assert "chunks are the machines in name order: 1/2 = brandons-macbook-pro, 2/2 = btm-nova" in text
    _, _, warn = plan.resolve_chunk("btm-nova=3,brandons-macbook-pro=1", (0, 2), this="btm-nova")
    assert "brandons-macbook-pro's share" in warn[0]


def test_equal_weights_printed_split_matches_what_runs():
    """Equal weights are the modulo split in jobs (check_chunk); split(), explain() and chunk_plan() count
    the same molecules (they used the weighted bins and printed other counts)."""
    ids = IDS[:1000]
    chunk, w, warn = plan.resolve_chunk("1,1", (0, 2))
    runs = [i for i in ids if jobs.in_chunk(i, jobs.check_chunk(chunk))]
    assert runs == [i for i in ids if jobs.chunk_index(i, 2) == 0]
    assert plan.split(ids, w)["1"] == runs
    assert f"this machine ({machine_name()}, chunk 1/2, weight 1): {len(runs):,} of 1,000 molecules" in \
        plan.chunk_plan(w, chunk, ids, warn)
    assert all(plan.assign(i, [2.0, 2.0, 2.0]) == jobs.chunk_index(i, 3) for i in ids)
    assert all(plan.assign(i, [3, 1]) == plan.weighted_index(i, [3, 1]) for i in ids)
    # a machine with weight 0 makes the weights unequal: weighted bins in jobs and in the plan alike
    w0 = (1.0, 1.0, 0.0)
    assert [i for i in ids if jobs.in_chunk(i, jobs.check_chunk((0, 3, w0)))] == plan.split(ids, dict(zip("abc", w0)))["a"]


def test_auto_split_is_stored_and_reused_by_the_other_machines():
    """--chunk-weights auto: the first machine stores the resolved split for the list; another machine that
    resolves auto later (after a re-benchmark changed the registry) reuses it, so the shares still add up."""
    ids = IDS[:3000]
    profiles = [prof("btm-nova", mmff_at_cpu=3000.0), prof("brandons-macbook-pro", mmff_at_cpu=1000.0)]
    a, wa, _ = plan.resolve_chunk("auto", None, "mmff", profiles, this="btm-nova", ids=ids)
    assert any("stored this split" in n for n in wa.notes)
    files = list(plan.splits_dir().glob("*.json"))
    assert len(files) == 1 and files[0].stem == plan.split_key("mmff", list(reversed(ids)) + ids[:5])
    profiles[1].throughput["mmff@cpu"] = 2500.0                           # the MacBook re-benchmarked
    b, wb, _ = plan.resolve_chunk("auto", None, "mmff", profiles, this="brandons-macbook-pro", ids=ids)
    assert b[2] == a[2] == (1000.0, 3000.0) and any("using the split stored for this list" in n for n in wb.notes)
    mine_a = {i for i in ids if jobs.in_chunk(i, jobs.check_chunk(a))}
    mine_b = {i for i in ids if jobs.in_chunk(i, jobs.check_chunk(b))}
    assert not mine_a & mine_b and mine_a | mine_b == set(ids)
    text = plan.chunk_plan(wb, b, ids, this="brandons-macbook-pro")
    assert "stored: benchmark mmff@cpu" in text and "--chunk-weights brandons-macbook-pro=1000,btm-nova=3000" in text
    # another list or method resolves (and stores) anew; explicit weights never use the store
    c, _, _ = plan.resolve_chunk("auto", None, "mmff", profiles, this="btm-nova", ids=ids[:10])
    assert c[2] == (2500.0, 3000.0)
    assert plan.resolve_chunk("btm-nova=1,brandons-macbook-pro=1", None, "mmff", profiles, this="btm-nova",
                              ids=ids)[0] == (1, 2, (1.0, 1.0))
    # SyncThing conflict copy (two machines stored at once): every machine takes the oldest and warns
    key = plan.split_key("mmff", ids)
    conflict = plan.splits_dir() / f"{key}.sync-conflict-20261105-000000-ABCDEFG.json"
    record = {**json.loads(files[0].read_text()), "weights": {"btm-nova": 1.0, "brandons-macbook-pro": 9.0},
              "created": "2000-01-01T00:00:00+00:00", "created_by": "brandons-macbook-pro"}
    conflict.write_text(json.dumps(record))
    d, wd, _ = plan.resolve_chunk("auto", None, "mmff", profiles, this="btm-nova", ids=ids)
    assert d[2] == (9.0, 1.0) and any("different splits were stored" in n for n in wd.notes)
    (plan.splits_dir() / f"{key}.json").write_text("{half a fi")      # unreadable files are ignored
    conflict.unlink()
    e, we, _ = plan.resolve_chunk("auto", None, "mmff", profiles, this="btm-nova", ids=ids)
    assert e[2] == (2500.0, 3000.0) and any("stored this split" in n for n in we.notes)


# ------------------------------------------------------------------------------------ weighted chunks in jobs
def test_weighted_chunk_helpers():
    assert jobs.check_chunk((0, 2, [3, 1])) == (0, 2, (3.0, 1.0))
    assert jobs.check_chunk((1, 2, (2.5, 2.5))) == (1, 2)          # equal weights = the plain modulo split
    assert jobs.check_chunk((1, 3)) == (1, 3) and jobs.check_chunk(None) is None
    for bad in ((0, 2, (1,)), (0, 2, (0, 1)), (0, 2, (1, -1)), (0, 2, ("a", 1)), (0, 2, 1, 1)):
        with pytest.raises(ValueError):
            jobs.check_chunk(bad)
    assert jobs.chunk_label((0, 2, (3.0, 1.0))) == "chunk1of2-75pct" and jobs.chunk_label((1, 3)) == "chunk2of3"
    assert jobs.chunk_label((1, 2, [3, 1])) == "chunk2of2-25pct"
    assert jobs.chunk_share((0, 2, (3, 1))) == 0.75 and jobs.chunk_share((0, 4)) == 0.25
    assert jobs.same_chunk((0, 2, (3, 1)), (0, 2, [6, 2])) and not jobs.same_chunk((0, 2, (3, 1)), (0, 2, (2, 1)))
    assert not jobs.same_chunk((0, 2, (3, 1)), (0, 2)) and jobs.same_chunk((0, 2), (0, 2, (1, 1)))
    assert not jobs.same_chunk(None, (0, 2)) and jobs.same_chunk(None, None)
    # unweighted chunks are exactly the old modulo split (existing run folders depend on it)
    for i in IDS[:500]:
        assert jobs.in_chunk(i, (1, 3)) == (jobs.chunk_index(i, 3) == 1)
        assert jobs.in_chunk(i, (0, 2, (3, 1))) == (plan.weighted_index(i, [3, 1]) == 0)


@pytest.fixture
def fake_run(monkeypatch):
    """pipeline.run_molecule that returns at once (the split, not the chemistry, is under test)."""
    calls = []

    def fake(inp, *a, **k):
        calls.append(inp.id)
        return pipeline.MoleculeResult(input=inp, status="ok", method="mmff")

    monkeypatch.setattr(pipeline, "run_molecule", fake)
    monkeypatch.setattr(export, "write_molecule", lambda *a, **k: [])
    return calls


def _cfg():
    return pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1,
                                gpu="off", write_files=False)


def test_weighted_chunks_in_runs_resume_and_mismatch(tmp_path, fake_run):
    inputs = [from_smiles(s, id=i) for i, s in MOLS]
    weights = (3.0, 1.0)
    for k in range(2):
        jobs.run_resumable(inputs, _cfg(), tmp_path / f"c{k}", chunk=(k, 2, weights))
    assert sorted(fake_run) == sorted(i for i, _ in MOLS)               # each molecule exactly once
    for k in range(2):
        st = jobs.open_run(tmp_path / f"c{k}")
        assert jobs.check_chunk(st.chunk) == (k, 2, weights)
        assert st.total == sum(plan.weighted_index(i, weights) == k for i, _ in MOLS)
        assert f"chunk{k + 1}of2-{75 if k == 0 else 25}pct" in st.summary()
    # resume without a chunk keeps the stored weights; the same split written differently is accepted
    fake_run.clear()
    jobs.run_resumable(inputs, _cfg(), tmp_path / "c0")
    jobs.run_resumable(inputs, _cfg(), tmp_path / "c0", chunk=(0, 2, (6, 2)))
    assert fake_run == []
    # other weights, or the unweighted chunk, would select other molecules: refused unless forced
    for other in ((0, 2, (1, 3)), (0, 2), (1, 2, weights)):
        with pytest.raises(jobs.ConfigMismatchError) as exc:
            jobs.run_resumable(inputs, _cfg(), tmp_path / "c0", chunk=other)
        assert "chunk" in exc.value.differences
    jobs.run_resumable(inputs, _cfg(), tmp_path / "c0", chunk=(0, 2, (1, 3)), force=True)
    assert jobs.check_chunk(jobs.open_run(tmp_path / "c0").chunk) == (0, 2, (1.0, 3.0))
    # an unweighted run folder is not changed by any of this
    jobs.run_resumable(inputs, _cfg(), tmp_path / "plain", chunk=(0, 2))
    with pytest.raises(jobs.ConfigMismatchError):
        jobs.run_resumable(inputs, _cfg(), tmp_path / "plain", chunk=(0, 2, weights))


def test_cli_build_with_chunk_weights(tmp_path, capsys, fake_run):
    mols = [f"{s} {i}" for i, s in MOLS]
    run = tmp_path / "run"
    base = ["build", *mols, "--method", "mmff", "--max-confs", "2", "--run-dir", str(run)]
    assert cli.main([*base, "--chunk", "1/2", "--chunk-weights", "3,1"]) == 0
    out = capsys.readouterr().out
    mine = sum(plan.weighted_index(i, [3, 1]) == 0 for i, _ in MOLS)
    assert f"this machine ({machine_name()}, chunk 1/2, weight 3): {mine} of 8 molecules" in out
    assert sorted(fake_run) == sorted(i for i, _ in MOLS if plan.weighted_index(i, [3, 1]) == 0)
    # resume keeps the split; another split of the same folder is refused
    assert cli.main(["build", "--resume", str(run)]) == 0
    assert cli.main(["build", "--resume", str(run), "--chunk", "1/2", "--chunk-weights", "1,3"]) == 2
    assert "chunk" in capsys.readouterr().err
    assert cli.main([*base, "--chunk-weights", "3,1"]) == 2                 # positional weights need --chunk
    assert "need --chunk" in capsys.readouterr().err


def test_cli_build_with_auto_weights(tmp_path, capsys, fake_run):
    me = machine_name()
    save_profile(prof(me, mmff_at_cpu=3000.0))
    save_profile(prof("zz-other", cores=4))                               # no benchmark: estimated
    mols = [f"{s} {i}" for i, s in MOLS]
    args = ["build", *mols, "--method", "mmff", "--run-dir", str(tmp_path / "run"), "--chunk-weights", "auto"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert f"this machine ({me}, weight 3000)" in out and "estimate: 4 cores" in out
    assert "Give the other machines the same split: --chunk-weights" in out
    assert cli.main([*args[:-1], "auto", "--method", "auto"]) == 2
