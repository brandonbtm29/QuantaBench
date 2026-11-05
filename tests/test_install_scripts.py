"""The one-command installers (scripts/install_*.sh, scripts/install_windows.ps1) and the launchers.

Dry runs go into a copy of the repository in a folder with a space (like "GitHub Projects") and must not
create anything. Real installs are not run here (they download PyTorch); see docs/INSTALL.md.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BASH_SCRIPTS = [SCRIPTS / "install_linux.sh", SCRIPTS / "install_macos.sh", REPO / "QuantaBench.command"]
PS1 = SCRIPTS / "install_windows.ps1"
BASH = shutil.which("bash") if sys.platform != "win32" else None
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def copy_repo(tmp_path: Path) -> Path:
    dst = tmp_path / "GitHub Projects" / "QuantaBench"
    dst.mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE", "QuantaBench.command", "QuantaBench.bat"):
        shutil.copy2(REPO / name, dst / name)
    for folder in ("envs", "scripts"):
        shutil.copytree(REPO / folder, dst / folder)
    return dst


def run_script(cmd: list[str], tmp_path: Path, **env) -> subprocess.CompletedProcess:
    environ = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environ.update(XDG_DATA_HOME=str(tmp_path / "xdg"), **env)
    return subprocess.run(cmd, capture_output=True, text=True, env=environ, timeout=120, stdin=subprocess.DEVNULL)


# --------------------------------------------------------------------------------------- bash: static
@pytest.mark.parametrize("script", BASH_SCRIPTS, ids=lambda p: p.name)
def test_bash_scripts_are_portable_to_macos(script):
    """macOS has bash 3.2 and BSD tools: no GNU-only flags, no bash 4 features."""
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!")
    assert "\r" not in text
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for pattern, why in [
        (r"\bsed\s+-i\b", "BSD sed -i needs a suffix argument"),
        (r"\breadlink\s+-f\b", "no readlink -f on older macOS"),
        (r"\bgrep\s+-\w*P", "BSD grep has no -P"),
        (r"\b(mapfile|readarray)\b", "bash 4"),
        (r"\bdeclare\s+-A\b", "bash 4 associative arrays"),
        (r"\$\{\w+(,,|\^\^)\}", "bash 4 case conversion"),
        (r"&>>", "bash 4"),
        (r"\bprintf\s+%q\b|\bprintf\s+'[^']*%q", "%q output differs between bash versions"),
    ]:
        assert not re.search(pattern, code), f"{script.name}: {why}"


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("script", BASH_SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax_and_shellcheck(script):
    assert subprocess.run([BASH, "-n", str(script)], capture_output=True).returncode == 0
    shellcheck = shutil.which("shellcheck")
    if shellcheck:
        cp = subprocess.run([shellcheck, "-s", "bash", str(script)], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="file modes")
def test_launchers_and_scripts_are_executable():
    for script in BASH_SCRIPTS:
        assert os.access(script, os.X_OK), f"{script.name} must be executable (git update-index --chmod=+x)"


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("script", ["install_linux.sh", "install_macos.sh"])
def test_help_and_bad_options(script, tmp_path):
    cp = run_script([BASH, str(SCRIPTS / script), "--help"], tmp_path)
    assert cp.returncode == 0 and "--with-engines" in cp.stdout and "--dry-run" in cp.stdout
    cp = run_script([BASH, str(SCRIPTS / script), "--no-such-option"], tmp_path)
    assert cp.returncode != 0 and "unknown option" in cp.stderr
    cp = run_script([BASH, str(SCRIPTS / script), "--settings-dir"], tmp_path)
    assert cp.returncode != 0 and "needs a value" in cp.stderr


# ------------------------------------------------------------------------------------- bash: dry runs
def _native_bash_script():
    if BASH is None:
        return None
    return {"linux": "install_linux.sh", "darwin": "install_macos.sh"}.get(sys.platform)


@pytest.mark.skipif(_native_bash_script() is None, reason="needs Linux or macOS")
def test_dry_run_changes_nothing(tmp_path):
    repo = copy_repo(tmp_path)
    settings = tmp_path / "SyncThing" / "QuantaBench" / "Settings"
    # `true` stands in for conda: its `env list --json` lists nothing, so both environments are "created"
    cmd = [BASH, str(repo / "scripts" / _native_bash_script()), "--dry-run", "--python", sys.executable,
           "--settings-dir", str(settings), "--with-engines", "--conda", shutil.which("true")]
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    cp = run_script(cmd, tmp_path)
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "dry run, nothing is changed" in out
    assert "-m venv" in out and "pip install -e '.[all,dev]'" in out
    assert "settings --set-dir" in out and "doctor" in out
    assert "envs/tools.yml" in out and "envs/psi4.yml" in out
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before   # nothing written
    if sys.platform == "linux":
        assert "--index-url https://download.pytorch.org/whl/cpu" in out
        assert "would write" in out and "quantabench.desktop" in out
        assert 'Exec="' + str(repo / ".venv" / "bin" / "quantabench") + '"' in out
    else:
        assert "QuantaBench.app" in out


@pytest.mark.skipif(_native_bash_script() is None, reason="needs Linux or macOS")
def test_foreign_venv_is_left_alone(tmp_path):
    """A Windows .venv (e.g. a checkout on /mnt/c seen from WSL) must not get a Linux venv on top."""
    repo = copy_repo(tmp_path)
    (repo / ".venv" / "Scripts").mkdir(parents=True)
    (repo / ".venv" / "pyvenv.cfg").write_text("home = C:\\Python312\n", encoding="utf-8")
    cp = run_script([BASH, str(repo / "scripts" / _native_bash_script()), "--python", sys.executable], tmp_path)
    assert cp.returncode != 0 and "another system" in cp.stderr
    assert not (repo / ".venv" / "bin").exists()


# ---------------------------------------------------------------------------------------- PowerShell
def test_powershell_script_is_ascii_and_5_1_safe():
    """Windows PowerShell 5.1 reads a BOM-less script as ANSI, and mangles double quotes in native
    arguments, so the script stays ASCII and its python -c snippets avoid double quotes."""
    raw = PS1.read_bytes()
    assert all(b < 128 for b in raw), "non-ASCII character in install_windows.ps1"
    text = raw.decode("ascii")
    snippets = re.findall(r'^\$(\w+Code) = "(.*)"\r?$', text, flags=re.M)
    assert len(snippets) >= 4
    for name, code in snippets:
        assert '\\"' not in code and '`"' not in code, name
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("??", "&&", "||"):                    # PowerShell 7 operators
        assert forbidden not in code, forbidden


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")
def test_powershell_script_parses():
    cmd = ("$e = $null; [System.Management.Automation.Language.Parser]::ParseFile('"
           + str(PS1).replace("'", "''") + "', [ref]$null, [ref]$e) | Out-Null; $e.Count; $e | ForEach-Object { $_.Message }")
    cp = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True,
                        timeout=120)
    assert cp.stdout.strip().splitlines()[0] == "0", cp.stdout + cp.stderr


@pytest.mark.skipif(sys.platform != "win32" or POWERSHELL is None, reason="needs Windows")
def test_powershell_dry_run_changes_nothing(tmp_path):
    repo = copy_repo(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    cp = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                         str(repo / "scripts" / "install_windows.ps1"), "-DryRun", "-Python", sys.executable,
                         "-SettingsDir", str(tmp_path / "Settings"), "-NoShortcut"],
                        capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "dry run, nothing is changed" in out and "pip install -e '.[all,dev]'" in out
    assert "checkout path has a space" in out and "settings --set-dir" in out
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


# ------------------------------------------------------------------------------------- docs and URLs
def _script_options(script: Path) -> set[str]:
    text = script.read_text(encoding="utf-8")
    if script.suffix == ".ps1":
        block = text.split("param(", 1)[1].split("\n)", 1)[0]
        return {"-" + m for m in re.findall(r"\$(\w+)", block)}
    return set(re.findall(r"^\s+(--[a-z-]+)\)", text, flags=re.M))


@pytest.mark.parametrize("doc", ["docs/INSTALL.md", "README.md"])
def test_documented_install_commands_use_real_options(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    found = 0
    for line in text.splitlines():
        m = re.search(r"scripts[/\\](install_(?:linux|macos)\.sh|install_windows\.ps1)(.*)", line)
        if not m:
            continue
        found += 1
        script = SCRIPTS / m.group(1)
        assert script.exists(), line
        options = _script_options(script)
        flags = re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", m.group(2))
        flags = [f for f in flags if f != "-File"]
        for flag in flags:
            assert flag in options, f"{doc}: {flag} is not an option of {script.name}"
    assert found >= 3


def test_install_commands_brandon_asked_for():
    text = (REPO / "docs" / "INSTALL.md").read_text(encoding="utf-8")
    assert ("cd ~/GitHub\\ Projects && git clone https://github.com/brandonbtm29/QuantaBench.git && cd QuantaBench && "
            "bash scripts/install_macos.sh --with-engines --settings-dir ~/SyncThing/QuantaBench/Settings") in text
    assert ('cd "$HOME\\GitHub Projects\\QuantaBench"; git pull; powershell -ExecutionPolicy Bypass -File '
            'scripts\\install_windows.ps1 -WithEngines -SettingsDir "$HOME\\SyncThing\\QuantaBench\\Settings"') in text


def test_miniforge_urls_match_its_readme():
    """URLs and silent-install flags from github.com/conda-forge/miniforge (README, checked 2026-11-04)."""
    unix = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    for name in ("install_linux.sh", "install_macos.sh"):
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert unix in text and '-b -p "$MINIFORGE_DIR"' in text
    text = PS1.read_text(encoding="ascii")
    assert "releases/latest/download/Miniforge3-Windows-x86_64.exe" in text
    assert '"/S", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/D=$MiniforgeDir"' in text


def test_windows_launcher_hides_the_console():
    bat = (REPO / "QuantaBench.bat").read_bytes().decode("ascii")
    assert 'start "" "%QB_VENV%\\Scripts\\pythonw.exe" -m quantabench' in bat
    assert "install_windows.ps1" in bat
    text = PS1.read_text(encoding="ascii")
    assert 'Scripts\\quantabench.exe' in text            # pip's GUI launcher (pythonw), not quantabench-cli
