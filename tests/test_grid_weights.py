"""``grid build --weights``: the same rules as ``build --chunk-weights`` (dist.plan), per grid target."""

import pytest

from quantabench.dist import client, plan, profile


def _ref(name, prof=None):
    info = {"profile": prof.to_dict()} if prof is not None else {}
    return client.WorkerRef(name=name, url=f"http://127.0.0.1:1/{name}", source="option", info=info)


def _prof(name, cores=8, throughput=None, engines=None):
    return profile.MachineProfile(name=name, physical_cores=cores, logical_cores=2 * cores,
                                  throughput=dict(throughput or {}), engines=dict(engines or {}))


@pytest.fixture
def no_registry(monkeypatch):
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: [])
    monkeypatch.setattr(profile, "machine_name", lambda: "this-box")
    monkeypatch.setattr(profile, "local_profile", lambda *a, **k: _prof("this-box", cores=4))


def test_positional_and_named(no_registry):
    refs = {"btm-nova": _ref("btm-nova")}
    names = ["btm-nova", client.LOCAL]
    assert client.resolve_weights("3,1", names, "mmff", refs) == {"btm-nova": 3.0, "local": 1.0}
    # named: "local" or this machine's own name, any order and spelling plan.norm_name accepts
    assert client.resolve_weights("this-box=1, BTM NOVA=2", names, "mmff", refs) == {"btm-nova": 2.0, "local": 1.0}
    assert client.resolve_weights("local=1,btm-nova=0", names, "mmff", refs) == {"btm-nova": 0.0, "local": 1.0}
    for bad, what in (("3", "need 2 weights"), ("btm-nova=1", "no weight for local"),
                      ("btm-nova=1,mac=1", "not in --machines"), ("0,0", "positive"), ("x,1", "not a number")):
        with pytest.raises(client.GridError, match=what):
            client.resolve_weights(bad, names, "mmff", refs)


def test_auto_uses_reported_profiles_estimates_and_unavailable(no_registry):
    logs = []
    nova = _prof("btm-nova", cores=12, throughput={"gfn1-xtb@cpu": 250.0})
    mac = _prof("mac", cores=10)                                   # no benchmark: core-count estimate
    gone = _prof("old-box", engines={"gfn1-xtb": False})           # cannot run the method
    refs = {"btm-nova": _ref("btm-nova", nova), "mac": _ref("mac", mac), "old-box": _ref("old-box", gone)}
    w = client.resolve_weights("auto", ["btm-nova", "mac", "old-box"], "gfn1-xtb", refs, logs.append)
    assert w["btm-nova"] == 250.0 and w["old-box"] == 0.0
    assert w["mac"] == plan.auto_weight(mac, "gfn1-xtb")[0] > 0
    assert any("estimate" in m for m in logs)
    # a target without any profile gets the mean of the others
    refs["bare"] = _ref("bare")
    w = client.resolve_weights("auto", ["btm-nova", "bare"], "gfn1-xtb", refs, logs.append)
    assert w == {"btm-nova": 250.0, "bare": 250.0}
    with pytest.raises(client.GridError, match="explicit --method"):
        client.resolve_weights("auto", ["btm-nova"], "auto", refs)
    with pytest.raises(client.GridError, match="can run"):
        client.resolve_weights("auto", ["old-box"], "gfn1-xtb", refs)


def test_auto_local_and_registry_fallback(monkeypatch, no_registry):
    stored = _prof("btm-nova", throughput={"mmff@cpu": 9000.0})
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: [stored])
    refs = {"btm-nova": _ref("btm-nova")}                          # the worker reported no profile
    w = client.resolve_weights("auto", ["btm-nova", client.LOCAL], "mmff", refs)
    assert w["btm-nova"] == 9000.0
    assert w["local"] == plan.auto_weight(_prof("this-box", cores=4), "mmff")[0]
    # nothing known anywhere: equal weights
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: [])
    monkeypatch.setattr(profile, "local_profile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert client.resolve_weights("auto", ["btm-nova", client.LOCAL], "mmff", refs) == {"btm-nova": 1.0, "local": 1.0}
