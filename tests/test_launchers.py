"""quantabench.launchers and `quantabench-cli shortcuts`: launchers with the app icon that hide the terminal (DIST-1).

The macOS bundle and the Linux .desktop file are plain files, so they are checked on every platform. The Windows
shortcut needs PowerShell and Windows: there one is made in a temporary folder and read back; elsewhere the
script's arguments are checked.
"""

import json
import os
import plistlib
import shutil
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from quantabench import cli, launchers


def fake_venv(repo: Path, name: str, platform: str = sys.platform) -> Path:
    venv = repo / name
    py = launchers.Variant(venv).python(platform)
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return venv


# ---------------------------------------------------------------------------------------------- icons
def test_icons_are_there_and_well_formed():
    png = (launchers.ASSETS / "quantabench.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (256, 256)
    ico = (launchers.ASSETS / "quantabench.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", ico[:6])
    sizes = sorted((ico[6 + 16 * i] or 256) for i in range(count))
    assert (reserved, kind) == (0, 1) and sizes[0] == 16 and sizes[-1] == 256 and {32, 48} <= set(sizes)
    icns = (launchers.ASSETS / "quantabench.icns").read_bytes()
    assert icns[:4] == b"icns" and struct.unpack(">I", icns[4:8])[0] == len(icns)
    kinds, pos = [], 8
    while pos < len(icns):
        kinds.append(icns[pos:pos + 4])
        pos += struct.unpack(">I", icns[pos + 4:pos + 8])[0]
    assert pos == len(icns) and {b"icp4", b"ic07", b"ic08", b"ic09", b"ic10"} <= set(kinds)
    assert "<svg" in (launchers.ASSETS / "quantabench.svg").read_text(encoding="utf-8")
    for platform in ("win32", "darwin", "linux"):
        assert launchers.icon_path(platform).is_file()


def test_window_icon(qapp):
    icon = launchers.window_icon()
    assert not icon.isNull() and icon.availableSizes()


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------------------------------- variants
def test_variants_of_a_checkout(tmp_path):
    repo = tmp_path / "QuantaBench"
    fake_venv(repo, ".venv-rocm")
    fake_venv(repo, ".venv")
    fake_venv(repo, ".venv-test")
    (repo / ".venv-broken").mkdir()                        # no pyvenv.cfg, no interpreter
    other = "linux" if sys.platform == "win32" else "win32"
    fake_venv(repo, ".venv-foreign", platform=other)       # e.g. a Windows venv seen from WSL
    vs = launchers.find_variants(repo)
    assert [v.prefix.name for v in vs] == [".venv", ".venv-rocm", ".venv-test"]
    assert [v.label for v in vs] == ["QuantaBench", "QuantaBench (AMD GPU)", "QuantaBench (test)"]
    assert [v.app_id for v in vs] == [launchers.APP_ID, launchers.APP_ID + ".rocm", launchers.APP_ID + ".test"]
    # no environment in the checkout: the running interpreter's
    assert launchers.find_variants(tmp_path / "empty") == [launchers.Variant(Path(sys.prefix),
                                                                             launchers.suffix_of(sys.prefix))]


def test_suffix_and_app_side(monkeypatch):
    assert launchers.suffix_of("/x/.venv") == "" and launchers.suffix_of("C:/x/.venv-rocm") == "rocm"
    assert launchers.suffix_of("/opt/conda") == "" and launchers.suffix_of("/x/.venv-") == ""
    monkeypatch.setattr(sys, "prefix", "/repo/.venv-rocm")
    assert launchers.desktop_file_name() == "quantabench-rocm"
    monkeypatch.setattr(sys, "prefix", "/repo/.venv")
    assert launchers.desktop_file_name() == "quantabench"
    assert launchers.set_app_id("linux") is False and launchers.set_app_id("darwin") is False
    v = launchers.Variant(Path("C:/qb/.venv"))
    assert v.python("win32").name == "pythonw.exe" and v.python("win32", windowed=False).name == "python.exe"
    assert v.python("darwin") == Path("C:/qb/.venv/bin/python")


# ----------------------------------------------------------------------------------------------- Linux
@pytest.mark.parametrize(("arg", "quoted"), [
    ("/usr/bin/python", "/usr/bin/python"),
    ("/home/me/GitHub Projects/QB/.venv/bin/python", '"/home/me/GitHub Projects/QB/.venv/bin/python"'),
    ('/a "b"/$c`d\\e', '"/a \\"b\\"/\\$c\\`d\\\\e"'),
    ("/100%/py", "/100%%/py"),
])
def test_desktop_exec_quoting(arg, quoted):
    assert launchers._desktop_quote(arg) == quoted


@pytest.mark.skipif(os.name == "nt", reason="POSIX paths in Exec=")
def test_linux_desktop_entry_and_remove(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(launchers, "_desktop_dir", lambda home, platform: home / "Desktop")
    repo = tmp_path / "GitHub Projects" / "QuantaBench"
    fake_venv(repo, ".venv", platform="linux")
    fake_venv(repo, ".venv-rocm", platform="linux")
    home = tmp_path / "home"
    variants = launchers.find_variants(repo, "linux")
    assert launchers.create(repo, platform="linux", home=home, dry_run=True, variants=variants) == [
        f"would write {home / '.local/share/applications/quantabench.desktop'}",
        f"would write {home / '.local/share/applications/quantabench-rocm.desktop'}"]
    assert not home.exists()
    made = launchers.create(repo, desktop=True, platform="linux", home=home, variants=variants)
    apps = home / ".local" / "share" / "applications"
    assert made == [str(apps / "quantabench.desktop"), str(home / "Desktop" / "quantabench.desktop"),
                    str(apps / "quantabench-rocm.desktop"), str(home / "Desktop" / "quantabench-rocm.desktop")]
    text = (apps / "quantabench.desktop").read_text(encoding="utf-8")
    entry = dict(line.split("=", 1) for line in text.splitlines()[1:] if "=" in line)
    assert text.startswith("[Desktop Entry]\n") and entry["Name"] == "QuantaBench" and entry["Terminal"] == "false"
    assert entry["Exec"] == f'"{repo / ".venv" / "bin" / "python"}" -m quantabench'
    assert entry["Path"] == str(repo) and Path(entry["Icon"]).name == "quantabench.png" and entry["StartupWMClass"]
    assert "Name=QuantaBench (AMD GPU)" in (apps / "quantabench-rocm.desktop").read_text(encoding="utf-8")
    if os.name != "nt":
        assert (apps / "quantabench.desktop").stat().st_mode & stat.S_IXUSR
    validate = shutil.which("desktop-file-validate")
    if validate:
        cp = subprocess.run([validate, str(apps / "quantabench.desktop")], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stdout + cp.stderr
    removed = launchers.create(repo, desktop=True, platform="linux", home=home, variants=variants, remove=True)
    assert len(removed) == 4 and not any(apps.iterdir())


# ----------------------------------------------------------------------------------------------- macOS
def test_macos_app_bundle(tmp_path):
    repo = tmp_path / "GitHub Projects" / "Quanta'Bench"
    venv = fake_venv(repo, ".venv", platform="darwin")
    home = tmp_path / "home"
    v = launchers.Variant(venv)
    made = launchers.create(repo, platform="darwin", home=home, variants=[v])
    app = home / "Applications" / "QuantaBench.app"
    assert made == [str(app)]
    with open(app / "Contents" / "Info.plist", "rb") as fh:
        info = plistlib.load(fh)
    assert info["CFBundleIdentifier"] == launchers.BUNDLE_ID and info["CFBundleExecutable"] == "QuantaBench"
    assert info["CFBundleIconFile"] == "QuantaBench.icns" and info["CFBundlePackageType"] == "APPL"
    assert (app / "Contents" / "Resources" / "QuantaBench.icns").read_bytes()[:4] == b"icns"
    exe = app / "Contents" / "MacOS" / "QuantaBench"
    script = exe.read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\n") and 'exec "$PY" -m quantabench' in script and "/opt/homebrew/bin" in script
    if os.name != "nt":
        assert exe.stat().st_mode & stat.S_IXUSR
    bash = shutil.which("bash") if os.name != "nt" else None
    if bash:
        assert subprocess.run([bash, "-n", str(exe)]).returncode == 0
        # the quoted paths come back as written (a quote in the folder name)
        out = subprocess.run([bash, "-c", script.split("export PATH")[0] + 'printf "%s\\n%s" "$REPO" "$PY"'],
                             capture_output=True, text=True).stdout
        assert out == f"{repo}\n{venv / 'bin' / 'python'}"
    # a second environment gets its own bundle and id; --desktop a copy on the Desktop
    rocm = launchers.Variant(fake_venv(repo, ".venv-rocm", platform="darwin"), "rocm")
    made = launchers.create(repo, desktop=True, platform="darwin", home=home, variants=[v, rocm])
    assert [Path(p).name for p in made] == ["QuantaBench.app", "QuantaBench.app", "QuantaBench (AMD GPU).app",
                                            "QuantaBench (AMD GPU).app"]
    with open(home / "Desktop" / "QuantaBench (AMD GPU).app" / "Contents" / "Info.plist", "rb") as fh:
        assert plistlib.load(fh)["CFBundleIdentifier"] == launchers.BUNDLE_ID + ".rocm"
    launchers.create(repo, desktop=True, platform="darwin", home=home, variants=[v, rocm], remove=True)
    assert not app.exists() and not (home / "Desktop" / "QuantaBench.app").exists()


# --------------------------------------------------------------------------------------------- Windows
def test_windows_spec_and_script(tmp_path, monkeypatch):
    repo = Path("C:/Users/me/GitHub Projects/QuantaBench")
    vs = [launchers.Variant(repo / ".venv"), launchers.Variant(repo / ".venv-rocm", "rocm")]
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["spec"] = json.loads(Path(cmd[cmd.index("-Spec") + 1]).read_text(encoding="utf-8"))
        seen["script"] = Path(cmd[cmd.index("-File") + 1]).read_bytes()
        assert kw["timeout"] and kw["stdin"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(cmd, 0, "C:\\Start\\QuantaBench.lnk\r\nC:\\Desk\\QuantaBench.lnk\r\n", "")

    monkeypatch.setattr(launchers, "_powershell", lambda: "powershell.exe")
    out = launchers.create(repo, desktop=True, platform="win32", variants=vs, run=fake_run)
    assert out == ["C:\\Start\\QuantaBench.lnk", "C:\\Desk\\QuantaBench.lnk"]
    assert seen["cmd"][:5] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
    assert seen["script"].startswith(b"\xef\xbb\xbf")             # BOM: PowerShell 5.1 reads UTF-8
    spec = seen["spec"]
    assert [(s["place"], s["name"]) for s in spec] == [("Programs", "QuantaBench.lnk"), ("Desktop", "QuantaBench.lnk"),
                                                       ("Programs", "QuantaBench (AMD GPU).lnk"),
                                                       ("Desktop", "QuantaBench (AMD GPU).lnk")]
    assert spec[0]["target"].endswith("pythonw.exe") and spec[0]["arguments"] == "-m quantabench"
    assert spec[0]["workdir"] == str(repo) and spec[0]["icon"].endswith("quantabench.ico")
    assert (spec[0]["appid"], spec[2]["appid"]) == (launchers.APP_ID, launchers.APP_ID + ".rocm")
    assert not spec[0]["remove"] and not spec[0]["dryrun"]
    assert "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3" in launchers.WINDOWS_SCRIPT     # the AppUserModelID property
    assert all(ord(c) < 128 for c in launchers.WINDOWS_SCRIPT)
    monkeypatch.setattr(launchers, "_powershell", lambda: None)
    with pytest.raises(RuntimeError, match="PowerShell was not found"):
        launchers.create(repo, platform="win32", variants=vs, run=fake_run)


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows")
def test_windows_shortcut_is_made_and_read_back(tmp_path):
    repo = tmp_path / "GitHub Projects" / "QuantaBench"
    v = launchers.Variant(fake_venv(repo, ".venv"))
    (v.prefix / "Scripts" / "pythonw.exe").write_text("", encoding="utf-8")
    folder = tmp_path / "Start Menu"
    folder.mkdir()
    out = launchers.create(repo, platform="win32", variants=[v], windows_folder=folder)
    lnk = folder / "QuantaBench.lnk"
    assert out == [str(lnk)] and lnk.is_file()
    ps = launchers._powershell()
    read = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($args[0]); "
            "$s.TargetPath; $s.Arguments; $s.WorkingDirectory; $s.IconLocation; "
            "$f = (New-Object -ComObject Shell.Application).NameSpace($args[1]).ParseName($args[2]); "
            "$f.ExtendedProperty('System.AppUserModel.ID')")
    cp = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", f"& {{ {read} }}", str(lnk), str(folder),
                         lnk.name], capture_output=True, text=True, timeout=120)
    lines = cp.stdout.splitlines()
    assert lines[0].lower() == str(v.python("win32")).lower(), cp.stdout + cp.stderr
    assert lines[1] == "-m quantabench" and lines[2].lower() == str(repo).lower()
    assert lines[3].lower().startswith(str(launchers.icon_path("win32")).lower())
    if len(lines) > 4 and lines[4]:                                 # the shell reports the taskbar id
        assert lines[4] == launchers.APP_ID
    launchers.create(repo, platform="win32", variants=[v], windows_folder=folder, remove=True)
    assert not lnk.exists()


# ------------------------------------------------------------------------------------------------- CLI
def test_cli_shortcuts_dry_run(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(launchers, "create", lambda repo, **kw: calls.append(kw) or ["would write /x/quantabench.desktop"])
    assert cli.main(["shortcuts", "--dry-run", "--desktop"]) == 0
    out = capsys.readouterr().out
    assert "would write /x/quantabench.desktop" in out and calls[0]["dry_run"] and calls[0]["desktop"]
    monkeypatch.setattr(launchers, "create", lambda repo, **kw: (_ for _ in ()).throw(RuntimeError("no shell")))
    assert cli.main(["shortcuts"]) == 1 and "no shell" in capsys.readouterr().out
