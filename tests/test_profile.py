"""Machine profiles, the machine registry and the throughput benchmark (dist.profile)."""

import json
import os
import time

import pytest

from quantabench import __version__, cli, resources
from quantabench import settings as settings_mod
from quantabench.chem import pipeline
from quantabench.dist import profile
from quantabench.dist.profile import MachineProfile


def full_profile(name="btm-nova", updated="2026-11-04T21:00:00+00:00") -> MachineProfile:
    return MachineProfile(
        name=name, os="Windows 11", arch="AMD64", python="3.12.10", quantabench_version="0.3.0.dev0",
        cpu="AMD Ryzen 9 7900X 12-Core Processor", physical_cores=12, logical_cores=24, ram_gb=63.2,
        gpus=[{"index": 0, "name": "AMD Radeon RX 9060 XT", "memory_gb": 15.9, "backend": "rocm", "arch": "gfx1200"}],
        torch="2.9.1+rocm7.2.1", engines={"mmff": True, "gfn1-xtb": True, "gfn2-xtb": False},
        programs={"xtb": "6.7.1", "crest": "", "mopac": "unknown"}, throughput={"mmff@cpu": 40000.0,
                                                                             "gfn1-xtb@cpu": 1500.0},
        address="http://100.64.0.1:8765", updated=updated, commit="bd1f22b893a3", torch_backend="rocm",
        wsl_programs=["crest"], benchmarks={"mmff@cpu": {"molecules": 50, "seconds": 4.5}}, notes=["a note"])


def test_profile_json_round_trip():
    p = full_profile()
    text = json.dumps(p.to_dict())
    assert MachineProfile.from_dict(json.loads(text)) == p
    # unknown keys are ignored, wrong types fall back to the default, a name is required
    d = {**p.to_dict(), "future_field": 1, "physical_cores": "12", "throughput": {"mmff@cpu": "fast", "x@cpu": 5},
         "gpus": [{"index": 0}, "junk"], "engines": {"mmff": True, "bad": "yes"}, "ram_gb": 64}
    q = MachineProfile.from_dict(d)
    assert q.physical_cores == 0 and q.throughput == {"x@cpu": 5.0} and q.gpus == [{"index": 0}]
    assert q.engines == {"mmff": True} and q.ram_gb == 64.0
    for bad in ({}, {"name": ""}, {"name": 3}, []):
        with pytest.raises((ValueError, TypeError)):
            MachineProfile.from_dict(bad)


def test_throughput_for():
    p = full_profile()
    p.throughput["gfn1-xtb@cuda:0"] = 2500.0
    assert profile.throughput_for(p, "gfn1-xtb") == 2500.0                # auto = fastest device
    assert profile.throughput_for(p, "gfn1-xtb", "cpu") == 1500.0
    assert profile.throughput_for(p, "aimnet2") is None and profile.throughput_for(p, "mmff", "cuda:0") is None
    assert profile.best_throughput(p, "gfn1-xtb") == ("gfn1-xtb@cuda:0", 2500.0)


def test_registry_tolerates_stale_partial_and_corrupt_files():
    d = profile.registry_dir()
    assert profile.load_profiles() == []                                  # no folder yet
    profile.save_profile(full_profile("btm-nova", "2026-11-04T21:00:00+00:00"))
    profile.save_profile(full_profile("brandons-macbook-pro", "2026-09-01T08:00:00+00:00"))   # stale
    (d / "garbage.json").write_text("{not json", encoding="utf-8")
    (d / "list.json").write_text("[1, 2]", encoding="utf-8")
    (d / "half.json").write_text('{"name": "half", "throughput": {"mmff@cpu": 10}, "updated": 5}', encoding="utf-8")
    (d / "nameless.json").write_text('{"physical_cores": 4}', encoding="utf-8")       # name from the file
    (d / "btm-nova.json.tmp").write_text("{", encoding="utf-8")                        # interrupted write
    newer = full_profile("btm-nova", "2026-11-05T01:00:00+00:00")
    newer.cpu = "newer copy"
    conflict = d / "btm-nova.sync-conflict-20261105-010000-ABCDEFG.json"             # SyncThing conflict
    conflict.write_text(json.dumps(newer.to_dict()), encoding="utf-8")
    profiles = profile.load_profiles()
    names = [p.name for p in profiles]
    assert names[:2] == ["btm-nova", "brandons-macbook-pro"] and sorted(names) == sorted(
        ["btm-nova", "brandons-macbook-pro", "half", "nameless"])
    assert profiles[0].cpu == "newer copy"                                # newest file of a machine wins
    half = next(p for p in profiles if p.name == "half")
    assert half.throughput == {"mmff@cpu": 10.0} and half.updated == "" and profile.profile_age_s(half) is None
    assert next(p for p in profiles if p.name == "nameless").physical_cores == 4
    assert profile.load_profile("brandons-macbook-pro").programs["xtb"] == "6.7.1"


def test_machine_name(monkeypatch):
    # the hostname path (Windows, Linux); on a Mac the Bonjour name comes first (test below), so leave it out here
    monkeypatch.setattr(profile, "_macos_local_hostname", lambda: "")
    monkeypatch.setattr(profile.socket, "gethostname", lambda: "BTM-NOVA.tail1234.ts.net")
    monkeypatch.setattr(resources, "is_wsl", lambda: False)
    assert profile.machine_name() == "btm-nova"
    monkeypatch.setattr(resources, "is_wsl", lambda: True)                # WSL2 shares the Windows hostname
    assert profile.machine_name() == "btm-nova-wsl"
    monkeypatch.setattr(profile.socket, "gethostname", lambda: "Brandons-MacBook-Pro.local")
    monkeypatch.setattr(resources, "is_wsl", lambda: False)
    assert profile.machine_name() == "brandons-macbook-pro"


def test_local_profile_keeps_measurements_and_takes_the_worker_address():
    p = profile.local_profile()
    assert p.name == profile.machine_name() and p.quantabench_version == __version__
    assert p.engines["mmff"] is True and set(p.programs) == {"xtb", "crest", "gxtb", "mopac", "psi4"}
    assert p.logical_cores >= 1 and p.ram_gb > 0 and profile.profile_age_s(p) < 60
    assert MachineProfile.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    old = MachineProfile(name=p.name, throughput={"mmff@cpu": 123.0}, benchmarks={"mmff@cpu": {"seconds": 1.0}},
                         address="http://old:8765")
    profile.save_profile(old)
    p = profile.local_profile()
    assert p.throughput == {"mmff@cpu": 123.0} and p.address == "http://old:8765"
    assert profile.local_profile(keep_previous=False).throughput == {}
    s = settings_mod.load()
    s.dist.workers[p.name] = "http://100.64.0.2:8765"
    settings_mod.save(s)
    assert profile.local_profile().address == "http://100.64.0.2:8765"


def test_git_commit_reads_worktrees_and_packed_refs(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo / ".git" / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/main\n", encoding="utf-8")
    assert profile.git_commit(repo) == "a" * 12
    (repo / ".git" / "refs" / "heads" / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    assert profile.git_commit(repo) == "b" * 12
    wt = tmp_path / "wt"
    gitdir = repo / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")                  # detached
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert profile.git_commit(wt) == "c" * 12
    assert profile.git_commit(tmp_path / "nothing") == ""


def test_benchmark_mmff_is_fast_and_sane(monkeypatch):
    monkeypatch.setattr(profile, "MIN_SECONDS", 1.0)
    for var in resources.THREAD_ENV_VARS:      # the benchmark sets thread caps; restore them afterwards
        monkeypatch.setenv(var, os.environ.get(var, "")) if var in os.environ else monkeypatch.delenv(var, raising=False)
    details = {}
    t = time.perf_counter()
    rates = profile.benchmark(["mmff"], ["cpu"], max_seconds=20, threads=1, details=details)
    assert time.perf_counter() - t < 30
    assert set(rates) == {"mmff@cpu"} and 100 < rates["mmff@cpu"] < 5e6, rates
    info = details["mmff@cpu"]
    assert info["molecules"] >= len(profile.BENCH_SET) and info["seconds"] >= 1.0 and not info["error"]
    assert info["set"] == "pfas-5" and info["per_hour"] == pytest.approx(rates["mmff@cpu"], rel=1e-3)
    # GPU devices are skipped for engines that cannot use them; unknown and unavailable methods are skipped
    assert profile.benchmark(["mmff", "no-such-method"], ["cuda:0"], max_seconds=5) == {}


def test_benchmark_is_cancellable_and_rejects_fallbacks(monkeypatch):
    def fake(inp, cfg, device="cpu", threads=1, should_stop=lambda: False, *a, **k):
        if should_stop():
            return pipeline.MoleculeResult(input=inp, status="cancelled")
        return pipeline.MoleculeResult(input=inp, status="ok", method=cfg.method, device="cpu")

    from quantabench.chem import engines

    monkeypatch.setattr(pipeline, "run_molecule", fake)
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))     # CI has no torch/dxtb
    monkeypatch.setattr(profile, "MIN_SECONDS", 0.0)
    from quantabench.chem import torch_compat

    monkeypatch.setattr(torch_compat, "dxtb_device", lambda dev: (str(dev), ""))       # dxtb would use the GPU
    details = {}
    assert profile.benchmark(["gfn1-xtb"], ["cuda:0"], details=details) == {}          # ran on the CPU instead
    assert details["gfn1-xtb@cuda:0"]["error"].startswith("fell back to cpu")
    # where QuantaBench keeps dxtb on the CPU (MPS, NOVA's ROCm wheel) the GPU is not timed at all
    monkeypatch.setattr(torch_compat, "dxtb_device", lambda dev: ("cpu", "dxtb ran on the CPU instead of mps"))
    details = {}
    assert profile.benchmark(["gfn1-xtb"], ["mps"], details=details) == {}
    assert details["gfn1-xtb@mps"]["skipped"] and not details["gfn1-xtb@mps"]["error"]
    assert set(profile.benchmark(["gfn1-xtb"], ["cpu"])) == {"gfn1-xtb@cpu"}
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    details = {}
    assert profile.benchmark(["mmff", "gfn1-xtb"], ["cpu"], should_stop=stop, details=details) == {}
    assert details["mmff@cpu"]["error"] == "cancelled" and "gfn1-xtb@cpu" not in details


def test_benchmark_with_lanes_goes_through_the_job_table(monkeypatch):
    from quantabench.chem import engines
    from quantabench.dist.lanes import Lane

    monkeypatch.setattr(pipeline, "run_molecule", lambda inp, cfg, *a, **k: pipeline.MoleculeResult(
        input=inp, status="ok", method=cfg.method))
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    details = {}
    rates = profile.benchmark(["mmff"], lanes=[Lane("cpu0", "cpu", 1), Lane("cpu1", "cpu", 1)], details=details)
    assert list(rates) == ["mmff@lanes:cpu:2x1"] and details["mmff@lanes:cpu:2x1"]["molecules"] == 5


def test_local_profile_bench_stores_throughput(monkeypatch):
    monkeypatch.setattr(profile, "benchmark", lambda methods, devices=None, details=None, **k: (
        details.update({f"{m}@cpu": {"molecules": 5, "seconds": 1.0} for m in methods}) or
        {f"{m}@cpu": 100.0 for m in methods}))
    profile.save_profile(MachineProfile(name=profile.machine_name(), throughput={"mmff@cpu": 1.0, "x@cpu": 2.0}))
    p = profile.local_profile(bench=True, methods=["mmff"])
    assert p.throughput == {"mmff@cpu": 100.0, "x@cpu": 2.0}            # re-measured wins, others are kept


def test_machines_cli(capsys, monkeypatch):
    assert cli.main(["machines"]) == 0
    assert "empty" in capsys.readouterr().out
    me = profile.machine_name()
    profile.save_profile(full_profile("zz-old-mac", "2026-01-01T00:00:00+00:00"))
    assert cli.main(["machines", "--refresh"]) == 0
    out = capsys.readouterr().out
    assert f"{me} (this machine)" in out and "zz-old-mac" in out and "days old" in out
    assert "gfn1-xtb@cpu 1,500, mmff@cpu 40,000 molecules/hour" in out and "xtb 6.7.1" in out
    assert "crest" in out and "worker:     http://100.64.0.1:8765" in out and "not measured" in out
    assert (profile.registry_dir() / f"{me}.json").is_file()
    assert cli.main(["machines", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["this_machine"] == me and {m["name"] for m in data["machines"]} == {me, "zz-old-mac"}
    assert all(m["age_s"] is not None for m in data["machines"])
    assert cli.main(["machines", "--bench", "nonsense"]) == 2
    # --bench measures and saves (the benchmark itself is tested above)
    monkeypatch.setattr(profile, "benchmark", lambda methods, devices=None, details=None, **k: {"mmff@cpu": 9000.0})
    assert cli.main(["machines", "--bench", "mmff"]) == 0
    assert "mmff@cpu 9,000" in capsys.readouterr().out


def test_doctor_json_includes_the_profile(capsys, monkeypatch):
    assert cli.main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["profile"]["name"] == profile.machine_name() and "engines" in data["profile"]
    assert "cpu_name" in data                                              # the hardware report is unchanged
    monkeypatch.setattr(profile, "benchmark", lambda methods, devices=None, details=None, **k: {"mmff@cpu": 7.0})
    assert cli.main(["doctor", "--json", "--bench"]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["profile"]["throughput"] == {"mmff@cpu": 7.0} and "Benchmarking" in captured.err
    assert profile.load_profile(profile.machine_name()).throughput == {"mmff@cpu": 7.0}


def test_machine_names_are_folder_safe_and_short():
    from quantabench.dist import worker

    long = "sjc22-bt149-7e2e954b-e1e6-4932-b73d-ceb1546a53b8-2a200862263a"     # a GitHub macOS runner
    name = profile.safe_machine_name(long)
    assert len(name) <= profile.MAX_MACHINE_NAME and name.startswith("sjc22-bt149-")
    assert name != profile.safe_machine_name(long[:-1] + "b")                 # still unique
    assert worker.check_label(name + "-wsl") and worker.check_label(name)
    assert profile.safe_machine_name("BTM NOVA") == "btm-nova"
    assert profile.safe_machine_name("Brandon's MacBook Pro") == "brandon-s-macbook-pro"
    assert profile.safe_machine_name("---") == "localhost"


# ------------------------------------------------------------------ benchmark sets and solvent-specific keys
def test_bench_method_keeps_water_and_gas_apart():
    assert profile.bench_method("tblite-gfn2", "water") == "tblite-gfn2+water"
    assert profile.bench_method("tblite-gfn2", "h2o") == "tblite-gfn2+water"          # canonical name
    assert profile.bench_method("aimnet2", "water") == "aimnet2"          # no water model: it runs in gas phase
    assert profile.bench_method("auto", "water") == "auto"
    assert profile.split_method("tblite-gfn2+water") == ("tblite-gfn2", "water")
    p = MachineProfile(name="m", throughput={"tblite-gfn2@cpu": 100.0, "tblite-gfn2+water@cpu": 50.0,
                                             "tblite-gfn2+water@lanes:cpu:6x2": 400.0})
    assert profile.throughput_for(p, "tblite-gfn2", "cpu") == 100.0
    assert profile.throughput_for(p, "tblite-gfn2+water", "cpu") == 50.0
    assert profile.best_throughput(p, "tblite-gfn2") == ("tblite-gfn2@cpu", 100.0)
    assert profile.best_throughput(p, "tblite-gfn2+water") == ("tblite-gfn2+water@lanes:cpu:6x2", 400.0)
    assert profile.throughput_for(p, "tblite-gfn1", "cpu") is None
    # the readers of the grid split (plan.auto_weight, used by client.resolve_weights and --chunk-weights)
    from quantabench.dist import plan

    assert plan.auto_weight(p, "tblite-gfn2+water") == (400.0, "benchmark tblite-gfn2+water@lanes:cpu:6x2")
    assert plan.auto_weight(p, "tblite-gfn2") == (100.0, "benchmark tblite-gfn2@cpu")
    p.engines = {"tblite-gfn2": False}
    assert plan.auto_weight(p, "tblite-gfn2+water")[0] == 0.0
    q = MachineProfile(name="q", physical_cores=8, throughput={"tblite-gfn2@cpu": 100.0})
    assert plan.auto_weight(q, "tblite-gfn2+water")[1].startswith("estimate")     # gas numbers are not used


def test_benchmark_records_solvent_and_set(monkeypatch):
    from quantabench.chem import engines

    ran = []

    def fake(inp, cfg, device="cpu", threads=1, should_stop=lambda: False, *a, **k):
        ran.append(inp.id)
        return pipeline.MoleculeResult(input=inp, status="ok", method=cfg.method, device="cpu", solvent=cfg.solvent)

    monkeypatch.setattr(pipeline, "run_molecule", fake)
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    monkeypatch.setattr(profile, "MIN_SECONDS", 0.0)
    details = {}
    rates = profile.benchmark(["tblite-gfn2", "aimnet2"], ["cpu"], solvent="water", details=details,
                              bench_set="mixed")
    assert list(rates) == ["tblite-gfn2+water@cpu"]
    info = details["tblite-gfn2+water@cpu"]
    assert (info["solvent"], info["set"]) == ("water", "mixed-10")
    assert {"cholesterol", "caffeine", "PFOA"} <= set(ran)
    assert details["aimnet2+water@?"]["error"] == "no implicit water model"     # not recorded as a gas number
    p = MachineProfile(name="m", throughput=rates, benchmarks=details)
    assert profile.set_of_key(p, "tblite-gfn2+water@cpu") == "mixed"
    assert profile.set_of_key(p, "mmff@cpu") == "pfas"
    # a run that fell back to gas phase gives no water number
    monkeypatch.setattr(pipeline, "run_molecule", lambda inp, cfg, *a, **k: pipeline.MoleculeResult(
        input=inp, status="ok", method=cfg.method, device="cpu", solvent=""))
    details = {}
    assert profile.benchmark(["tblite-gfn2"], ["cpu"], solvent="water", details=details) == {}
    assert details["tblite-gfn2+water@cpu"]["error"].endswith("ran in gas phase instead of water")
    with pytest.raises(ValueError, match="unknown benchmark set"):
        profile.bench_molecules("big")


def test_machines_cli_passes_bench_set_and_solvent(monkeypatch):
    seen = {}

    def fake_refresh(bench, methods, devices, max_seconds, wsl, out=None, solvent="", bench_set="pfas"):
        seen.update(solvent=solvent, bench_set=bench_set, methods=methods)
        return None, None

    monkeypatch.setattr(profile, "refresh_from_cli", fake_refresh)
    monkeypatch.setattr(profile, "_parse_methods", lambda text, engines: ["tblite-gfn2"])
    assert cli.main(["machines", "--bench", "tblite-gfn2", "--bench-set", "mixed", "--solvent", "water"]) == 130
    assert seen == {"solvent": "water", "bench_set": "mixed", "methods": ["tblite-gfn2"]}


def test_machine_name_on_macos_is_the_bonjour_name(monkeypatch):
    """On a Mac socket.gethostname() follows the network (on Tailscale: "Brandons-MBP.tail...ts.net"), which put
    one Mac in the registry under two names; the LocalHostName from System Settings is stable."""
    import socket
    import sys

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(socket, "gethostname", lambda: "Brandons-MBP.tail70150b.ts.net")
    monkeypatch.setattr(profile, "_macos_local_hostname", lambda: "Brandons-MacBook-Pro")
    assert profile.machine_name() == "brandons-macbook-pro"
    monkeypatch.setattr(profile, "_macos_local_hostname", lambda: "")          # scutil missing: the hostname
    assert profile.machine_name() == "brandons-mbp"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(profile, "_macos_local_hostname", lambda: "never-used")
    assert profile.machine_name() == "brandons-mbp"
