"""The 2D sketcher's files (quantabench.sketcher_assets) and `quantabench-cli sketcher`: download, sha256 check,
unpack, folder overrides. A small fake zip and a fake sha256 stand in for the release; nothing goes to the network."""

import hashlib
import io
import urllib.error
import zipfile

import pytest

from quantabench import settings as sm
from quantabench import sketcher_assets as sa

FILES = {"wasm_shell.html": b"<!doctype html><title>sketcher</title>", "qtloader.js": b"function qtLoad(){}",
         "Sketcher.js": b"var schrodinger_sketcher_entry;", "Sketcher.wasm": b"\0asm\1\0\0\0",
         "sketcher_loading.gif": b"GIF89a"}


def make_zip(path, files=FILES, prefix=""):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(prefix + name, data)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def opener_for(data, calls=None):
    def opener(req, timeout=None):
        if calls is not None:
            calls.append(req.full_url)
        return FakeResponse(data)
    return opener


def test_install_from_zip_checks_sha_and_writes_licence(tmp_path):
    z, sha = make_zip(tmp_path / "dl.zip")
    programs = tmp_path / "programs"
    folder = sa.install(programs=programs, zip_path=z, sha256=sha)
    assert folder == programs / sa.FOLDER and sa.is_complete(folder)
    assert (folder / "Sketcher.wasm").read_bytes() == FILES["Sketcher.wasm"]
    licence = (folder / sa.LICENSE_FILE).read_text(encoding="utf-8")
    assert "BSD 3-Clause" in licence and "Schrodinger, LLC" in licence
    assert sa.SHA256 in (folder / sa.SOURCE_FILE).read_text(encoding="utf-8")
    assert z.is_file()                                   # a zip given by hand is kept
    assert not list(programs.glob(".*tmp*"))             # no temporary folder left


def test_repository_keeps_the_licence_text():
    text = sa.LICENSE_COPY.read_text(encoding="utf-8")
    assert text.startswith("BSD 3-Clause License") and "Schrodinger, LLC" in text


def test_wrong_sha_is_refused_and_nothing_installed(tmp_path):
    z, _sha = make_zip(tmp_path / "dl.zip")
    with pytest.raises(sa.SketcherAssetError, match="not the expected file"):
        sa.install(programs=tmp_path / "p", zip_path=z, sha256="0" * 64)
    assert not (tmp_path / "p" / sa.FOLDER).exists()


def test_complete_folder_is_kept_unless_forced(tmp_path):
    z, sha = make_zip(tmp_path / "dl.zip")
    folder = sa.install(programs=tmp_path / "p", zip_path=z, sha256=sha)
    (folder / "marker").write_text("x")
    assert sa.install(programs=tmp_path / "p", opener=lambda *a, **k: pytest.fail("no download expected")) == folder
    assert (folder / "marker").exists()
    sa.install(programs=tmp_path / "p", zip_path=z, sha256=sha, force=True)
    assert not (folder / "marker").exists() and sa.is_complete(folder)


def test_download_checks_sha_reports_progress_and_deletes_zip(tmp_path):
    z, sha = make_zip(tmp_path / "src.zip")
    seen, calls = [], []
    folder = sa.install(programs=tmp_path / "p", sha256=sha, opener=opener_for(z.read_bytes(), calls),
                        progress=lambda d, t: seen.append((d, t)))
    assert sa.is_complete(folder) and calls == [sa.URL]
    assert seen and seen[-1] == (z.stat().st_size, z.stat().st_size)
    assert not (tmp_path / "p" / sa.ZIP_NAME).exists()   # a downloaded zip goes once unpacked
    assert not list((tmp_path / "p").glob("*.part"))


def test_download_with_wrong_sha_leaves_nothing(tmp_path):
    z, _sha = make_zip(tmp_path / "src.zip")
    with pytest.raises(sa.SketcherAssetError, match="does not have the pinned sha256"):
        sa.install(programs=tmp_path / "p", sha256="1" * 64, opener=opener_for(z.read_bytes()))
    assert not any((tmp_path / "p").iterdir())


def test_offline_gives_manual_instructions(tmp_path):
    def offline(req, timeout=None):
        raise urllib.error.URLError("Name or service not known")

    with pytest.raises(sa.SketcherAssetError) as err:
        sa.install(programs=tmp_path / "p", opener=offline)
    msg = str(err.value)
    assert sa.URL in msg and sa.SHA256 in msg and "--from-zip" in msg and str(tmp_path / "p") in msg
    assert not list((tmp_path / "p").glob("*.part"))


def test_cancelled_download(tmp_path):
    import threading

    z, sha = make_zip(tmp_path / "src.zip")
    stop = threading.Event()
    stop.set()
    with pytest.raises(sa.SketcherAssetError, match="cancelled"):
        sa.install(programs=tmp_path / "p", sha256=sha, opener=opener_for(z.read_bytes()), cancel=stop)
    assert not list((tmp_path / "p").glob("*.part"))


def test_zip_put_into_programs_folder_is_used(tmp_path):
    programs = tmp_path / "p"
    programs.mkdir()
    _z, sha = make_zip(programs / sa.ZIP_NAME)
    folder = sa.install(programs=programs, sha256=sha, opener=lambda *a, **k: pytest.fail("no download expected"))
    assert sa.is_complete(folder) and (programs / sa.ZIP_NAME).exists()


def test_zip_with_a_top_folder(tmp_path):
    z, sha = make_zip(tmp_path / "dl.zip", prefix="sketcher-2026-4/")
    folder = sa.install(programs=tmp_path / "p", zip_path=z, sha256=sha)
    assert (folder / "wasm_shell.html").is_file() and not (folder / "sketcher-2026-4").exists()


@pytest.mark.parametrize("name", ["../evil.js", "/abs/evil.js"])
def test_unsafe_zip_paths_are_refused(tmp_path, name):
    z, sha = make_zip(tmp_path / "dl.zip", files={**FILES, name: b"x"})
    with pytest.raises(sa.SketcherAssetError, match="refusing to unpack"):
        sa.install(programs=tmp_path / "p", zip_path=z, sha256=sha)
    assert not (tmp_path / "evil.js").exists()


def test_zip_without_the_editor(tmp_path):
    z, sha = make_zip(tmp_path / "dl.zip", files={"wasm_shell.html": b"x"})
    with pytest.raises(sa.SketcherAssetError, match="does not contain .*Sketcher.wasm"):
        sa.install(programs=tmp_path / "p", zip_path=z, sha256=sha)
    assert not (tmp_path / "p" / sa.FOLDER).exists()


def _folder(tmp_path, name="own"):
    folder = tmp_path / name
    folder.mkdir()
    for n, data in FILES.items():
        (folder / n).write_bytes(data)
    return folder


def test_locate_order(tmp_path, monkeypatch):
    programs = tmp_path / "p"
    s = sm.Settings()
    monkeypatch.delenv(sa.ENV_DIR, raising=False)
    loc = sa.locate(s, programs)
    assert not loc.ok and loc.source == "missing" and "not downloaded" in loc.problem
    assert "not downloaded" in sa.status_text(s, programs)
    z, sha = make_zip(tmp_path / "dl.zip")
    sa.install(programs=programs, zip_path=z, sha256=sha)
    loc = sa.locate(s, programs)
    assert loc.ok and loc.source == "download" and loc.path == programs / sa.FOLDER
    own = _folder(tmp_path)
    s.set_engine_path_for(sa.ENGINE_KEY, str(own))           # this machine's folder wins over the download
    assert sa.locate(s, programs).path == own and sa.locate(s, programs).source == "settings"
    env = _folder(tmp_path, "env")
    monkeypatch.setenv(sa.ENV_DIR, str(env))                 # the environment wins over both
    assert sa.locate(s, programs).path == env and "from QUANTABENCH_SKETCHER_DIR" in sa.status_text(s, programs)


def test_configured_folder_without_files_is_reported(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "wasm_shell.html").write_text("x")
    monkeypatch.setenv(sa.ENV_DIR, str(bad))
    loc = sa.locate(sm.Settings(), tmp_path / "p")
    assert not loc.ok and loc.source == "environment" and "Sketcher.wasm" in loc.problem
    assert "not usable" in sa.status_text(sm.Settings(), tmp_path / "p")


def test_programs_dir_is_the_crest_build_folder(monkeypatch):
    import sys

    if sys.platform == "darwin":       # install_macos.sh's PROGRAMS_DIR
        assert str(sa.programs_dir()).endswith("Library/Application Support/QuantaBench/programs")
    assert sa.release_dir().name == sa.FOLDER and sa.release_dir().parent.name == "programs"


def test_cli_install_from_zip_and_use(tmp_path, monkeypatch, capsys):
    from quantabench.cli import main

    monkeypatch.delenv(sa.ENV_DIR, raising=False)
    monkeypatch.setattr(sa, "programs_dir", lambda: tmp_path / "p")
    z, sha = make_zip(tmp_path / "dl.zip")
    monkeypatch.setattr(sa, "SHA256", sha)
    assert main(["sketcher"]) == 0
    assert "not downloaded" in capsys.readouterr().out
    assert main(["sketcher", "--install", "--from-zip", str(z)]) == 0
    out = capsys.readouterr().out
    assert "is ready in" in out and str(tmp_path / "p" / sa.FOLDER) in out
    assert main(["sketcher", "--use", str(tmp_path / "missing")]) == 1
    own = _folder(tmp_path)
    assert main(["sketcher", "--use", str(own)]) == 0
    assert "this machine's folder" in capsys.readouterr().out
    assert sa.locate().path == own                           # saved for this machine in the (test) settings
    assert main(["sketcher", "--use", ""]) == 0
    assert sa.locate().source == "download"
    assert main(["sketcher", "--from-zip", str(z)]) == 2     # --from-zip goes with --install


def test_webengine_problem_messages(monkeypatch):
    from quantabench.gui import sketcher_web as sw

    monkeypatch.delenv(sw.ENV_DISABLE, raising=False)
    monkeypatch.setattr(sw, "webengine_installed", lambda: False)
    assert "quantabench[editor]" in sw.problem() and "not installed" in sw.problem()
    monkeypatch.setattr(sw, "webengine_installed", lambda: True)
    monkeypatch.setitem(sw._state, "registered", False)
    monkeypatch.setitem(sw._state, "error", "")
    assert "Restart QuantaBench" in sw.problem()
    monkeypatch.setitem(sw._state, "error", "ImportError: libnss3.so: cannot open shared object file")
    assert "could not be loaded" in sw.problem() and "libnss3" in sw.problem()
    monkeypatch.setitem(sw._state, "registered", True)
    assert sw.problem() == ""
    monkeypatch.setenv(sw.ENV_DISABLE, "1")
    assert "turned off" in sw.problem()


def test_cli_install_reports_a_bad_zip(tmp_path, monkeypatch, capsys):
    from quantabench.cli import main

    monkeypatch.setattr(sa, "programs_dir", lambda: tmp_path / "p")
    z, _sha = make_zip(tmp_path / "dl.zip")
    assert main(["sketcher", "--install", "--from-zip", str(z)]) == 1
    assert "not the expected file" in capsys.readouterr().err
