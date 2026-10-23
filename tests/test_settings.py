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
