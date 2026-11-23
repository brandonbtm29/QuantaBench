"""App launchers that start QuantaBench without a terminal window, with its icon (DIST-1).

    quantabench-cli shortcuts [--desktop] [--remove] [--dry-run]

One launcher per Python environment of the checkout (``.venv``: "QuantaBench"; ``.venv-rocm``: "QuantaBench (AMD
GPU)"; any other ``.venv-NAME``: "QuantaBench (NAME)"), or for the running interpreter when the checkout has none:

- **Windows:** a Start menu shortcut (and one on the Desktop with ``--desktop``) to the environment's
  ``pythonw.exe -m quantabench``, started in the checkout, with ``quantabench.ico``. It carries the app's
  AppUserModelID (``app_id``), which the app also sets for itself (``set_app_id``), so the taskbar shows the
  QuantaBench icon instead of Python's, the running window groups with the shortcut, and a pinned taskbar icon
  starts the app. WScript.Shell cannot set that ID, so the shortcut is written through IShellLink and
  IPropertyStore by a short PowerShell script (the same way as Brandon's other lab apps).
- **macOS:** ``~/Applications/QuantaBench.app`` (Launchpad, Spotlight, the Dock), a small bundle whose executable
  is a shell script that runs the environment's ``python -m quantabench``, with ``quantabench.icns``. It is the
  bundle ``scripts/install_macos.sh`` made before (same place, same bundle id), now with the icon.
- **Linux:** ``~/.local/share/applications/quantabench.desktop`` (the application menu; WSLg also lists it in the
  Windows Start menu) with the PNG icon; ``--desktop`` copies it to the desktop folder.

The launchers point at the checkout, not at a copy, so an update (Help > Check for updates...) needs no new
launcher; make them again when the checkout moves. ``QuantaBench.bat`` and ``QuantaBench.command`` in the
repository still work (they flash a console or Terminal window for a moment). No Qt here except
``window_icon``.
"""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import __version__

APP_NAME = "QuantaBench"
BUNDLE_ID = "io.github.brandonbtm29.quantabench"     # macOS (scripts/install_macos.sh used the same)
APP_ID = "io.github.brandonbtm29.QuantaBench"        # Windows AppUserModelID; the app sets the same one
DESKTOP_ID = "quantabench"                           # Linux .desktop file name (QGuiApplication.desktopFileName)
COMMENT = "Molecular modeling workbench: conformers, ML potentials, xTB and QSAR descriptors"
ASSETS = Path(__file__).resolve().parent / "assets"
#: the label of a known environment suffix (``.venv-rocm`` -> "AMD GPU")
VARIANT_LABELS = {"rocm": "AMD GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


@dataclass(frozen=True)
class Variant:
    """One Python environment that runs QuantaBench."""
    prefix: Path                 # the environment (a venv folder, or sys.prefix)
    suffix: str = ""             # "" for .venv (or the running interpreter), "rocm" for .venv-rocm, ...

    @property
    def label(self) -> str:
        if not self.suffix:
            return APP_NAME
        return f"{APP_NAME} ({VARIANT_LABELS.get(self.suffix, self.suffix)})"

    @property
    def app_id(self) -> str:
        return app_id(self.suffix)

    def python(self, platform: str | None = None, windowed: bool = True) -> Path:
        """The interpreter that starts the GUI: ``pythonw.exe`` on Windows (no console)."""
        platform = platform or sys.platform
        if platform == "win32":
            return self.prefix / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
        return self.prefix / "bin" / "python"


def app_id(suffix: str = "") -> str:
    return f"{APP_ID}.{suffix}" if suffix else APP_ID


def suffix_of(prefix: str | os.PathLike) -> str:
    """``.venv-rocm`` -> "rocm"; ``.venv`` and anything else -> ""."""
    name = Path(prefix).name
    return name[len(".venv-"):] if name.startswith(".venv-") and len(name) > len(".venv-") else ""


def icon_path(platform: str | None = None) -> Path:
    platform = platform or sys.platform
    name = {"win32": "quantabench.ico", "darwin": "quantabench.icns"}.get(platform, "quantabench.png")
    return ASSETS / name


def find_variants(repo: Path | None, platform: str | None = None) -> list[Variant]:
    """The checkout's environments (``.venv`` first, then ``.venv-*`` by name) that have an interpreter for this
    platform; else the running interpreter's environment."""
    platform = platform or sys.platform
    found: list[Variant] = []
    if repo is not None and Path(repo).is_dir():
        names = sorted((p for p in Path(repo).iterdir() if p.name == ".venv" or p.name.startswith(".venv-")),
                       key=lambda p: (p.name != ".venv", p.name))
        for venv in names:
            v = Variant(venv, suffix_of(venv))
            if (venv / "pyvenv.cfg").is_file() and v.python(platform).is_file():
                found.append(v)
    if not found:
        found.append(Variant(Path(sys.prefix), suffix_of(sys.prefix)))
    return found


# ------------------------------------------------------------------------------------- the app's own side
def set_app_id(platform: str | None = None) -> bool:
    """Windows: tell the taskbar this process is QuantaBench (the variant of its environment), not Python. Call
    it before the first window opens. Returns True when it was set."""
    if (platform or sys.platform) != "win32":
        return False
    try:
        import ctypes

        return ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id(suffix_of(sys.prefix))) == 0
    except (AttributeError, OSError):
        return False


def desktop_file_name() -> str:
    """The .desktop id of this environment's launcher (Wayland matches the window to it by this name)."""
    suffix = suffix_of(sys.prefix)
    return f"{DESKTOP_ID}-{suffix}" if suffix else DESKTOP_ID


def window_icon():
    """The app icon as a QIcon (Qt imported here only)."""
    from PySide6.QtGui import QIcon

    icon = QIcon()
    for name in ("quantabench.png", "quantabench.svg"):
        path = ASSETS / name
        if path.is_file():
            icon.addFile(str(path))
    return icon


# --------------------------------------------------------------------------------------------- planning
@dataclass(frozen=True)
class Launcher:
    """One launcher file to write (or remove)."""
    path: Path | None            # None on Windows until PowerShell resolves the special folder
    kind: str                    # "windows" | "macos" | "linux"
    variant: Variant
    place: str = ""              # Windows: "Programs" or "Desktop"; macOS/Linux: "" or "Desktop"

    def describe(self) -> str:
        if self.path is not None:
            return str(self.path)
        where = "the Start menu" if self.place == "Programs" else "the Desktop"
        return f"{self.variant.label}.lnk in {where}"


def _desktop_dir(home: Path, platform: str) -> Path:
    if platform.startswith("linux"):
        try:
            cp = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=10)  # noqa: S607
            out = cp.stdout.strip()
            if cp.returncode == 0 and out and Path(out) != home:
                return Path(out)
        except (OSError, subprocess.SubprocessError):
            pass
    return home / "Desktop"


def plan(repo: Path | None, desktop: bool = False, platform: str | None = None, home: Path | None = None,
         variants: list[Variant] | None = None) -> list[Launcher]:
    platform = platform or sys.platform
    home = Path(home) if home is not None else Path.home()
    variants = variants if variants is not None else find_variants(repo, platform)
    out: list[Launcher] = []
    for v in variants:
        if platform == "win32":
            out.append(Launcher(None, "windows", v, "Programs"))
            if desktop:
                out.append(Launcher(None, "windows", v, "Desktop"))
        elif platform == "darwin":
            out.append(Launcher(home / "Applications" / f"{v.label}.app", "macos", v))
            if desktop:
                out.append(Launcher(home / "Desktop" / f"{v.label}.app", "macos", v, "Desktop"))
        else:
            name = f"{DESKTOP_ID}-{v.suffix}.desktop" if v.suffix else f"{DESKTOP_ID}.desktop"
            apps = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "applications"
            out.append(Launcher(apps / name, "linux", v))
            if desktop:
                out.append(Launcher(_desktop_dir(home, platform) / name, "linux", v, "Desktop"))
    return out


# ------------------------------------------------------------------------------------------------ macOS
def mac_script(repo: Path | None, variant: Variant) -> str:
    """The bundle's executable: run the environment's python -m quantabench in the checkout."""
    py = variant.python("darwin")
    workdir = repo if repo is not None else Path.home()
    return f"""#!/bin/bash
# Made by `quantabench-cli shortcuts` (quantabench/launchers.py): starts {variant.label} without a Terminal window.
# Run it again if the checkout moves.
REPO={shlex.quote(str(workdir))}
PY={shlex.quote(str(py))}
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"   # git for Help > Check for updates..., Homebrew tools
if [ ! -x "$PY" ]; then
  /usr/bin/osascript -e 'display alert "QuantaBench is not installed here" message "Run scripts/install_macos.sh in the QuantaBench folder."'
  exit 1
fi
cd "$REPO" || exit 1
exec "$PY" -m quantabench
"""


def mac_info_plist(variant: Variant) -> dict:
    bundle_id = BUNDLE_ID + (f".{variant.suffix}" if variant.suffix else "")
    return {
        "CFBundleName": variant.label,
        "CFBundleDisplayName": variant.label,
        "CFBundleIdentifier": bundle_id,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": "QuantaBench.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
    }


def _write_mac_app(app: Path, repo: Path | None, variant: Variant) -> None:
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    exe = contents / "MacOS" / APP_NAME
    exe.write_text(mac_script(repo, variant), encoding="utf-8", newline="\n")
    exe.chmod(0o755)
    shutil.copyfile(icon_path("darwin"), contents / "Resources" / "QuantaBench.icns")
    with open(contents / "Info.plist", "wb") as fh:
        plistlib.dump(mac_info_plist(variant), fh)
    os.utime(app)               # Finder picks up the new icon


# ------------------------------------------------------------------------------------------------ Linux
def _desktop_quote(arg: str) -> str:
    """One Exec= argument (the Desktop Entry spec: double quotes, with \\ " ` $ escaped; % doubled)."""
    arg = arg.replace("%", "%%")
    if not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    return '"' + "".join("\\" + c if c in '"`$\\' else c for c in arg) + '"'


def desktop_entry(repo: Path | None, variant: Variant, platform: str = "linux") -> str:
    py = variant.python(platform)
    workdir = repo if repo is not None else Path.home()
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={variant.label}",
        "GenericName=Molecular modeling workbench",
        f"Comment={COMMENT}",
        f"Exec={_desktop_quote(str(py))} -m quantabench",
        f"Path={workdir}",
        f"Icon={icon_path(platform)}",
        "Terminal=false",
        "Categories=Education;Science;Chemistry;",
        f"StartupWMClass={APP_NAME}",
        "StartupNotify=true",
        f"X-QuantaBench-Version={__version__}",
        "",
    ])


# ---------------------------------------------------------------------------------------------- Windows
#: Writes (or removes) shortcuts from a JSON list, through IShellLink + IPropertyStore: WScript.Shell cannot set
#: the AppUserModelID (System.AppUserModel.ID, {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3} 5) that ties the taskbar
#: button to the app. Windows PowerShell 5.1 compatible; prints each shortcut's path.
WINDOWS_SCRIPT = r"""
param([string]$Spec)
$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
namespace QuantaBenchShortcut {
  [ComImport, Guid("00021401-0000-0000-C000-000000000046")] class CShellLink {}
  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
  interface IShellLinkW {
    void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder f, int cch, IntPtr fd, uint flags);
    void GetIDList(out IntPtr pidl); void SetIDList(IntPtr pidl);
    void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetHotkey(out short k); void SetHotkey(short k);
    void GetShowCmd(out int c); void SetShowCmd(int c);
    void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch, out int i);
    void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string s, int i);
    void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string s, uint r);
    void Resolve(IntPtr hwnd, uint flags);
    void SetPath([MarshalAs(UnmanagedType.LPWStr)] string s);
  }
  [StructLayout(LayoutKind.Sequential, Pack = 4)]
  public struct PropertyKey { public Guid fmtid; public uint pid; }
  [StructLayout(LayoutKind.Explicit)] public struct PropVariant {
    [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(8)] public long pad;
  }
  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
  interface IPropertyStore {
    void GetCount(out uint c); void GetAt(uint i, out PropertyKey k);
    void GetValue(ref PropertyKey k, out PropVariant v);
    void SetValue(ref PropertyKey k, ref PropVariant v); void Commit();
  }
  public static class Lnk {
    public static void Write(string lnk, string target, string args, string dir, string icon, string desc,
                             string appId) {
      var link = (IShellLinkW)new CShellLink();
      link.SetPath(target); link.SetArguments(args); link.SetWorkingDirectory(dir);
      link.SetIconLocation(icon, 0); link.SetDescription(desc);
      var key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };
      var v = new PropVariant { vt = 31, p = Marshal.StringToCoTaskMemUni(appId) };
      var store = (IPropertyStore)link;
      try { store.SetValue(ref key, ref v); store.Commit(); } finally { Marshal.FreeCoTaskMem(v.p); }
      ((IPersistFile)link).Save(lnk, true);
    }
  }
}
"@
$items = Get-Content -Raw -Encoding UTF8 $Spec | ConvertFrom-Json
foreach ($it in $items) {
  if ($it.dir) { $folder = $it.dir } else { $folder = [Environment]::GetFolderPath($it.place) }
  $lnk = Join-Path $folder $it.name
  if ($it.remove) {
    if (Test-Path -LiteralPath $lnk) { Remove-Item -LiteralPath $lnk; Write-Output $lnk }
    continue
  }
  if ($it.dryrun) { Write-Output $lnk; continue }
  [QuantaBenchShortcut.Lnk]::Write($lnk, $it.target, $it.arguments, $it.workdir, $it.icon, $it.description, $it.appid)
  Write-Output $lnk
}
"""


def windows_spec(launchers: list[Launcher], repo: Path | None, remove: bool = False,
                 dry_run: bool = False, folder: Path | None = None) -> list[dict]:
    """What the PowerShell script gets for each shortcut (separate so it can be tested anywhere). ``folder``
    replaces the Start menu and Desktop folders (tests)."""
    workdir = str(repo if repo is not None else Path.home())
    return [{"place": la.place, "dir": str(folder) if folder is not None else "", "name": f"{la.variant.label}.lnk",
             "target": str(la.variant.python("win32")), "arguments": "-m quantabench", "workdir": workdir,
             "icon": str(icon_path("win32")), "description": COMMENT, "appid": la.variant.app_id, "remove": remove,
             "dryrun": dry_run}
            for la in launchers]


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def _windows(launchers: list[Launcher], repo: Path | None, remove: bool, dry_run: bool,
             run=subprocess.run, folder: Path | None = None) -> list[str]:
    ps = _powershell()
    if ps is None:
        raise RuntimeError("PowerShell was not found, so the shortcuts cannot be made")
    with tempfile.TemporaryDirectory(prefix="qb-shortcuts-") as tmp:
        script, spec = Path(tmp) / "shortcuts.ps1", Path(tmp) / "spec.json"
        script.write_text(WINDOWS_SCRIPT, encoding="utf-8-sig")   # BOM: Windows PowerShell 5.1 reads it as UTF-8
        spec.write_text(json.dumps(windows_spec(launchers, repo, remove, dry_run, folder)), encoding="utf-8")
        cp = run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script),  # noqa: S603
                  "-Spec", str(spec)], capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL,
                 creationflags=_NO_WINDOW)
    if cp.returncode != 0:
        raise RuntimeError("Windows could not make the shortcuts: " + ((cp.stderr or "") + (cp.stdout or "")).strip()[-600:])
    return [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------- do it
def create(repo: Path | None, desktop: bool = False, platform: str | None = None, home: Path | None = None,
           dry_run: bool = False, remove: bool = False, variants: list[Variant] | None = None,
           run=subprocess.run, windows_folder: Path | None = None) -> list[str]:
    """Write (``remove``: delete) the launchers of ``plan``; returns their paths ("would ..." lines with
    ``dry_run``). ``windows_folder`` replaces the Start menu and Desktop (tests). Raises RuntimeError or OSError
    with a message."""
    platform = platform or sys.platform
    launchers = plan(repo, desktop, platform, home, variants)
    if platform == "win32":
        paths = _windows(launchers, repo, remove, dry_run, run=run, folder=windows_folder)
        if dry_run:
            return [f"would {'remove' if remove else 'write'} {p}" for p in paths]
        return paths
    done: list[str] = []
    for la in launchers:
        if la.path is None:                 # only Windows launchers leave the path to PowerShell
            continue
        if remove:
            if la.path.exists() or la.path.is_symlink():
                if dry_run:
                    done.append(f"would remove {la.path}")
                elif la.path.is_dir():
                    shutil.rmtree(la.path)
                    done.append(str(la.path))
                else:
                    la.path.unlink()
                    done.append(str(la.path))
            continue
        if dry_run:
            done.append(f"would write {la.path}")
            continue
        if la.kind == "macos":
            _write_mac_app(la.path, repo, la.variant)
        else:
            la.path.parent.mkdir(parents=True, exist_ok=True)
            la.path.write_text(desktop_entry(repo, la.variant, platform), encoding="utf-8")
            la.path.chmod(0o755)
            if la.place == "":
                _refresh_menu(la.path.parent)
        done.append(str(la.path))
    return done


def _refresh_menu(apps: Path) -> None:
    exe = shutil.which("update-desktop-database")
    if exe:
        try:
            subprocess.run([exe, str(apps)], capture_output=True, timeout=30)  # noqa: S603
        except (OSError, subprocess.SubprocessError):
            pass


def what(platform: str | None = None) -> str:
    """What ``create`` makes on this platform, in a few words (the menu entry and the CLI's help)."""
    platform = platform or sys.platform
    if platform == "win32":
        return "Start menu shortcuts"
    if platform == "darwin":
        return "QuantaBench.app in ~/Applications"
    return "an application-menu entry"
