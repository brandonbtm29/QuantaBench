import json
from pathlib import Path

from quantabench import settings as sm


def test_roundtrip(isolated_settings):
    s = sm.load()
    s.resources.threads = 7
    s.build.ph = 5.5
    s.output_dir = "~/SyncThing/QuantaBench/Runs"
    path = sm.save(s)
    assert path.parent == isolated_settings / "settings"
    again = sm.load()
    assert again.resources.threads == 7 and again.build.ph == 5.5
    assert again.output_path() == Path.home() / "SyncThing" / "QuantaBench" / "Runs"


def test_unknown_keys_are_ignored(isolated_settings):
    folder = isolated_settings / "settings"
    folder.mkdir(parents=True)
    (folder / sm.SETTINGS_FILE).write_text(json.dumps({"resources": {"threads": 3, "bogus": 1}, "nope": 2}))
    s = sm.load()
    assert s.resources.threads == 3


def test_pointer_file(tmp_path, monkeypatch):
    monkeypatch.delenv(sm.ENV_SETTINGS_DIR, raising=False)
    place = tmp_path / "pointer_home"
    target = tmp_path / "synced" / "Settings"
    monkeypatch.setattr(sm, "pointer_locations", lambda: [place])
    monkeypatch.setattr(sm, "default_settings_dir", lambda: tmp_path / "default")
    assert sm.settings_dir() == tmp_path / "default"
    sm.save(sm.Settings())  # writes to default
    new = sm.set_settings_dir(target, pointer_place=place)
    assert new == target.resolve() or new == target
    assert (target / sm.SETTINGS_FILE).exists()  # copied over


def test_portable_home():
    p = Path.home() / "SyncThing" / "QuantaBench"
    assert sm.portable(p) == "~/SyncThing/QuantaBench"


def test_old_settings_get_new_build_defaults(isolated_settings):
    folder = isolated_settings / "settings"
    folder.mkdir(parents=True)
    (folder / sm.SETTINGS_FILE).write_text(json.dumps({"build": {"ph": 6.0, "refine_method": "gfn1-xtb"}}))
    s = sm.load()
    assert s.build.ph == 6.0 and s.build.batched == "auto" and s.build.frequencies is False
    from quantabench.chem.pipeline import BuildConfig

    s.build.frequencies, s.build.batched = True, "on"
    cfg = BuildConfig.from_settings(s)
    assert cfg.frequencies and cfg.batched == "on" and cfg.method == "gfn1-xtb"


def test_surface_settings_roundtrip_and_reach_the_build_config(isolated_settings):
    """Radii set, per-element overrides and the probe go through settings.json into BuildConfig."""
    from quantabench.chem.pipeline import BuildConfig, surface_options

    s = sm.Settings()
    assert (s.surfaces.radii_set, s.surfaces.sasa_probe_A, s.surfaces.radii_overrides) == ("bondi", 1.4, {})
    s.surfaces.radii_set = "spartan"
    s.surfaces.radii_overrides = {"C": 1.8, "F": 1.47}
    s.surfaces.sasa_probe_A = 1.0
    sm.save(s)
    back = sm.load()
    assert back.surfaces.radii_set == "spartan" and back.surfaces.radii_overrides == {"C": 1.8, "F": 1.47}
    cfg = BuildConfig.from_settings(back)
    opts = surface_options(cfg)
    assert opts.radii_set == "spartan" and dict(opts.overrides) == {6: 1.8, 9: 1.47} and opts.probe_A == 1.0
    assert sm.Settings.from_dict({"surfaces": {"radii_overrides": {"C": "x", "N": True, "O": 1.5}}}
                                 ).surfaces.radii_overrides == {"C": "x", "O": 1.5}


def test_engine_paths_per_machine(isolated_settings, monkeypatch):
    """One shared settings file: NOVA keeps its wsl: CREST, the Mac its own build, other machines the default."""
    monkeypatch.setattr(sm, "_this_machine", lambda: "Brandons-MacBook-Pro")
    s = sm.Settings()
    s.engines.paths = {"xtb": "~/bin/xtb"}                       # every machine
    s.set_engine_path_for("crest", "wsl:/home/b/micromamba/envs/qb-tools/bin/crest", "BTM-NOVA")
    s.set_engine_path_for("crest", "~/qb-programs/crest-macos-arm64/bin/crest")   # this machine
    s.set_engine_path_for("xtb", "~/bin/xtb")                    # same as the default: not stored again
    assert s.engines.machine_paths == {"btm-nova": {"crest": "wsl:/home/b/micromamba/envs/qb-tools/bin/crest"},
                                       "brandons-macbook-pro": {"crest": "~/qb-programs/crest-macos-arm64/bin/crest"}}
    assert s.engine_paths_for() == {"xtb": ("~/bin/xtb", "all machines"),
                                    "crest": ("~/qb-programs/crest-macos-arm64/bin/crest", "this machine")}
    assert s.engine_paths_for("btm_nova")["crest"][0].startswith("wsl:")       # machine names as in machine_lanes
    assert s.engine_paths_for("linux-box") == {"xtb": ("~/bin/xtb", "all machines")}
    s.set_engine_path_for("xtb", "", "BTM-NOVA")                 # "" = automatic on NOVA despite the default
    assert "xtb" not in s.engine_paths_for("btm-nova")
    sm.save(s)
    again = sm.load()                                             # survives the JSON round trip
    assert again.engines.machine_paths == s.engines.machine_paths
    assert again.engine_paths_for("btm-nova") == s.engine_paths_for("btm-nova")
    s.set_engine_path_for("crest", "")                            # back to automatic: the entry goes away
    assert "brandons-macbook-pro" not in s.engines.machine_paths


def test_bad_machine_paths_are_dropped(isolated_settings):
    folder = isolated_settings / "settings"
    folder.mkdir(parents=True)
    (folder / sm.SETTINGS_FILE).write_text(json.dumps({"engines": {
        "machine_paths": {"nova": {"crest": "wsl:crest", "bad": 3}, "mac": "not a table"},
        "paths": {"xtb": {"nested": "no"}, "crest": "/opt/crest"}}}))
    s = sm.load()
    assert s.engines.machine_paths == {"nova": {"crest": "wsl:crest"}}
    assert s.engines.paths == {"crest": "/opt/crest"}


def test_save_keeps_the_other_machines_entries(isolated_settings):
    """The settings file is shared: a GUI that loaded it hours ago (and saves at every Run) must not write back
    its old copy of another machine's CREST path or lanes, nor bring back an entry another machine removed."""
    s0 = sm.load()
    s0.set_lanes_for("auto", "btm-nova")
    s0.set_lanes_for("4", "old-laptop")
    sm.save(s0)
    nova = sm.load()                                    # NOVA's GUI, open all day
    mac = sm.load()                                     # meanwhile on the Mac:
    mac.set_engine_path_for("crest", "~/qb/crest-macos-arm64/bin/crest", "brandons-macbook-pro")
    mac.set_lanes_for("5", "brandons-macbook-pro")
    del mac.dist.machine_lanes["old-laptop"]
    sm.save(mac)
    nova.set_lanes_for("12", "btm-nova")                # NOVA changes its own lanes and saves its stale copy
    nova.resources.threads = 20
    sm.save(nova)
    again = sm.load()
    assert again.engines.machine_paths == {"brandons-macbook-pro": {"crest": "~/qb/crest-macos-arm64/bin/crest"}}
    assert again.dist.machine_lanes == {"btm-nova": "12", "brandons-macbook-pro": "5"}
    assert again.resources.threads == 20                # everything else: the saving process's values
    assert nova.engines.machine_paths == again.engines.machine_paths   # and its copy is current again
    nova.set_engine_path_for("crest", "", "brandons-macbook-pro")      # removing another machine's entry works
    sm.save(nova)
    assert "brandons-macbook-pro" not in sm.load().engines.machine_paths


def test_settings_made_in_code_are_written_as_they_are(isolated_settings):
    s = sm.load()
    s.set_lanes_for("auto", "btm-nova")
    sm.save(s)
    fresh = sm.Settings()                               # not from load(): no merge
    fresh.set_lanes_for("2", "brandons-macbook-pro")
    sm.save(fresh)
    assert sm.load().dist.machine_lanes == {"brandons-macbook-pro": "2"}


def test_undoing_monkeypatch_never_exposes_the_real_settings_folder(monkeypatch):
    """conftest points the settings at a session folder too, so code that saves after a test's
    ``monkeypatch.undo()`` (a window closed later) cannot reach the real settings.json (2026-11-07)."""
    from quantabench import settings as sm

    monkeypatch.undo()
    assert "quantabench-tests-" in str(sm.settings_dir())
