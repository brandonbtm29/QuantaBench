<#
.SYNOPSIS
Install or update QuantaBench on Windows 11. Safe to rerun: every step that is already done is skipped.

.DESCRIPTION
    powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 [options]

    -WithEngines        also set up the external programs: Miniforge (only if no conda, mamba or micromamba
                        is found), then the conda environments envs\tools-windows.yml (xtb, MOPAC) and
                        envs\psi4.yml (Psi4), created or updated
    -SettingsDir DIR    keep settings in DIR, e.g. "$HOME\SyncThing\QuantaBench\Settings"
    -Gpu cpu|rocm|cuda|auto
                        cpu (default): PyPI's PyTorch, which is the CPU build on Windows and the fastest
                        on the CPU. rocm: additionally a second environment .venv-rocm with AMD's ROCm
                        wheels for Radeon GPUs, and a second shortcut. cuda: NVIDIA's build in .venv.
                        auto: cpu for now (docs/dev/BACKLOG.md PERF-1)
    -ReplaceTorch       replace an installed PyTorch of a different build (default: keep it, or ask)
    -Python PATH        the Python 3.12 to create the environments with
    -Conda PATH         the conda, mamba or micromamba to use for -WithEngines
    -MiniforgeDir DIR   where to install Miniforge if needed (default %USERPROFILE%\miniforge3; no spaces)
    -DesktopShortcut    also put a shortcut on the Desktop (the Start Menu entry is always made)
    -NoShortcut         make no shortcuts
    -Reinstall          run pip install -e even when pyproject.toml has not changed
    -DryRun             print every action without doing it
#>
[CmdletBinding()]
param(
    [switch]$WithEngines,
    [string]$SettingsDir = "",
    [ValidateSet("cpu", "rocm", "cuda", "auto")][string]$Gpu = "cpu",
    [switch]$ReplaceTorch,
    [string]$Python = "",
    [string]$Conda = "",
    [string]$MiniforgeDir = "",
    [switch]$DesktopShortcut,
    [switch]$NoShortcut,
    [switch]$Reinstall,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# a trailing backslash would escape the closing quote when PowerShell 5.1 passes it to a program
$SettingsDir = $SettingsDir.TrimEnd("\", "/")
$Venv = Join-Path $Repo ".venv"
$VenvRocm = Join-Path $Repo ".venv-rocm"
$Extras = "all,dev"
# Not on PyPI for Windows (docs/PLATFORMS.md section 4): dxtb ported to tad-libcint 0.4 (GFN2, polarizability and IR
# intensities; FORK-1) and tblite wheels (in-process GFN2 with ALPB; FORK-2), both from Brandon's forks
$ForkPkgs = @("dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1", "tad-libcint>=0.4", "tblite")
$TbliteLinks = "https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1"
if (-not $MiniforgeDir) { $MiniforgeDir = Join-Path $env:USERPROFILE "miniforge3" }
$MiniforgeUrl = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
$TorchCudaIndex = "https://download.pytorch.org/whl/cu132"
# AMD's ROCm 7.2.1 wheels for Windows (Python 3.12 only); see docs/INSTALL.md for newer releases
$RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
$RocmSdk = @("$RocmBase/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
             "$RocmBase/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
             "$RocmBase/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
             "$RocmBase/rocm-7.2.1.tar.gz")
$RocmTorch = "$RocmBase/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"

# --------------------------------------------------------------------------------------------- helpers
function Step([string]$Text) { Write-Host ""; Write-Host "==> $Text" -ForegroundColor Cyan }
function Info([string]$Text) { Write-Host "    $Text" }
function Ok([string]$Text) { Write-Host "    [ok] $Text" -ForegroundColor Green }
function Warn([string]$Text) { Write-Host "    [!] $Text" -ForegroundColor Yellow }
function Fail([string]$Text) { Write-Host ""; Write-Host "ERROR: $Text" -ForegroundColor Red; exit 1 }

function Format-Arg([string]$Arg) {
    # quoted for copy-pasting into PowerShell
    if ($Arg -eq "" -or $Arg -match '[\s''"`$;&|<>(){}\[\],@#]') { return "'" + ($Arg -replace "'", "''") + "'" }
    return $Arg
}

function Show-Command([string]$Exe, [string[]]$Arguments) {
    $parts = @("&", (Format-Arg $Exe)) + @($Arguments | ForEach-Object { Format-Arg $_ })
    Write-Host ("    > " + ($parts -join " "))
}

function Run {
    # print a command, then run it unless -DryRun; a non-zero exit stops the installer with -FailMessage
    param([string]$Exe, [string[]]$Arguments = @(), [string]$FailMessage = "", [switch]$AllowFail)
    Show-Command $Exe $Arguments
    if ($DryRun) { return 0 }
    $ErrorActionPreference = "Continue"      # native programs writing to stderr is not an error (PS 5.1)
    & $Exe @Arguments | Out-Host
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFail) {
        $why = "command failed with exit code $code"
        if ($FailMessage) { $why = "$why. $FailMessage" }
        Fail $why
    }
    return $code
}

function Get-Output([string]$Exe, [string[]]$Arguments = @()) {
    # stdout of a read-only command, or $null when it cannot run or fails
    $ErrorActionPreference = "Continue"
    try { $out = & $Exe @Arguments 2>$null } catch { return $null }
    if ($LASTEXITCODE -ne 0) { return $null }
    return (($out | Out-String).Trim())
}

function Write-TextFile([string]$Path, [string]$Content) {
    if ((Test-Path $Path) -and ((Get-Content -Raw $Path).Trim() -eq $Content.Trim())) { Ok "$Path is up to date"; return }
    if ($DryRun) { Info "would write ${Path}:"; $Content -split "`n" | ForEach-Object { Info "    | $_" }; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
    [IO.File]::WriteAllText($Path, $Content + "`r`n", (New-Object Text.ASCIIEncoding))
    Info "wrote $Path"
}

# Python code passed to python -c must not contain double quotes: Windows PowerShell 5.1 does not escape
# them for native programs.
$PyVersionCode = "import sys; print('%d.%d' % sys.version_info[:2])"
$PyExeCode = "import sys; print(sys.executable)"
$TorchCode = "import importlib.metadata as m; print(m.version('torch'))"
$WhereCode = "import os, quantabench; print(os.path.realpath(quantabench.__file__))"
$StaleTadCode = "from importlib import metadata as m; print(' '.join(n for n in ('tad-dftd3', 'tad-dftd4') " +
                "if any((d.metadata['Name'] or '').lower().replace('_', '-') == n for d in m.distributions())))"

function Get-PyVersion([string]$Exe) { if (-not (Test-Path $Exe)) { return $null }; return Get-Output $Exe @("-c", $PyVersionCode) }
function Get-TorchVersion([string]$Exe) { if (-not (Test-Path $Exe)) { return "" }; $v = Get-Output $Exe @("-c", $TorchCode); if ($v) { return $v }; return "" }
function Get-TorchFlavor([string]$Version) {
    if ($Version -match "\+rocm") { return "rocm" }
    if ($Version -match "\+cu") { return "cuda" }
    return "cpu"                             # +cpu or no tag: PyPI's Windows build is the CPU build
}
# .NET instead of Get-FileHash: Windows PowerShell started from PowerShell 7 (as on GitHub's runners) inherits
# PowerShell 7's PSModulePath and then cannot load the module that provides Get-FileHash.
function Get-Sha256([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { $bytes = $sha.ComputeHash($stream) } finally { $stream.Dispose(); $sha.Dispose() }
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}
function Test-Interactive { try { return -not [Console]::IsInputRedirected } catch { return $false } }

# ------------------------------------------------------------------------------------------------ start
$mode = ""
if ($DryRun) { $mode = ": dry run, nothing is changed" }
Write-Host "QuantaBench installer (Windows)$mode"
Info "repository: $Repo"
Info ("Windows " + [Environment]::OSVersion.Version.ToString() + ", PowerShell " + $PSVersionTable.PSVersion.ToString())
if ($Gpu -eq "auto") {
    $Gpu = "cpu"
    Info "-Gpu auto: using the CPU build. No QuantaBench engine uses a GPU on Windows yet (BACKLOG GPU-1/GPU-2), and the CPU build is faster on the CPU (PERF-1). Use -Gpu rocm for a separate GPU environment."
}

Step "Windows checks"
if ($Repo -match " ") {
    Info "The checkout path has a space (e.g. 'GitHub Projects'). QuantaBench and pip handle it. Known side effects:"
    Info "  - AMD's ROCm PyTorch prints 'GitHub: Unknown command line argument ...offload-arch.exe' on import (harmless)"
    if ($WithEngines) { Info "  - Miniforge must not be installed in a folder with spaces; it goes to $MiniforgeDir" }
} else {
    Ok "no spaces in the checkout path"
}
$sac = $null
try {
    $sac = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -Name VerifiedAndReputablePolicyState -ErrorAction Stop).VerifiedAndReputablePolicyState
} catch { $sac = $null }
if ($sac -eq 1 -or $sac -eq 2) {
    $sacMode = "on"
    if ($sac -eq 2) { $sacMode = "in evaluation mode" }
    Warn "Smart App Control is $sacMode. It can block brand-new scientific DLLs (RDKit, VTK, PyTorch extensions):"
    Warn "  'DLL load failed ... An Application Control policy has blocked this file'. See docs\INSTALL.md, 'Smart App Control'."
} elseif ($sac -eq 0) {
    Ok "Smart App Control is off"
} else {
    Info "Smart App Control state unknown (no policy key)"
}
if (Get-Command git -ErrorAction SilentlyContinue) { Ok "git (needed by Help > Check for updates...)" }
else { Warn "git was not found: Help > Check for updates... needs it. Install it with: winget install Git.Git" }

# ----------------------------------------------------------------------------------------- 1. Python
Step "Python 3.12"
$VPy = Join-Path $Venv "Scripts\python.exe"
$Py = $null
function Find-Python {
    if ($Python) {
        $v = Get-PyVersion $Python
        if (-not $v) { Fail "-Python $Python does not run" }
        if ($v -ne "3.12") { Warn "-Python $Python is Python $v; AMD's ROCm wheels need 3.12" }
        return (Resolve-Path $Python).Path
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = Get-Output "py" @("-3.12", "-c", $PyExeCode)
        if ($exe -and (Test-Path $exe)) { return $exe }
    }
    $candidates = @()
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notmatch "WindowsApps") { $candidates += $onPath.Source }   # not the Store alias
    $candidates += @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
                     (Join-Path $env:ProgramFiles "Python312\python.exe"))
    foreach ($c in $candidates) { if ((Get-PyVersion $c) -eq "3.12") { return $c } }
    return $null
}
if ((Test-Path $VPy) -and -not ($Gpu -eq "rocm" -and -not (Test-Path (Join-Path $VenvRocm "Scripts\python.exe")))) {
    $v = Get-PyVersion $VPy
    if (-not $v) { Fail "$Venv is broken (its Python no longer runs). Remove the .venv folder and run this script again." }
    Ok ".venv exists (Python $v); no system Python needed"
} else {
    $Py = Find-Python
    if (-not $Py) {
        $msg = "Python 3.12 was not found. Install it, then run this script again:`n" +
               "      winget install Python.Python.3.12`n" +
               "  or the Windows installer for the newest 3.12.x from https://www.python.org/downloads/windows/`n" +
               "  (AMD's GPU wheels are built for 3.12 only.) Or pass -Python C:\path\to\python.exe"
        if (-not $DryRun) { Fail $msg }
        Warn $msg
        $Py = "py"
    } else {
        Ok "found $Py (Python $(Get-PyVersion $Py))"
    }
}

# ------------------------------------------------------------------------------- 2. environments
function New-Venv([string]$Dir) {
    $exe = Join-Path $Dir "Scripts\python.exe"
    if (Test-Path $exe) { Ok "$Dir"; return }
    if ((Test-Path (Join-Path $Dir "pyvenv.cfg")) -or (Test-Path (Join-Path $Dir "bin"))) {
        Fail "$Dir is a virtual environment for another system (no Scripts\python.exe), e.g. made in WSL. Leave it alone and use a separate clone."
    }
    if ($Py -eq "py") { Run "py" @("-3.12", "-m", "venv", $Dir) | Out-Null }
    else { Run $Py @("-m", "venv", $Dir) | Out-Null }
    Run $exe @("-m", "pip", "install", "--quiet", "--upgrade", "pip") | Out-Null
}
Step "Virtual environment (.venv)"
New-Venv $Venv

# ------------------------------------------------------------------------------------------ 3. PyTorch
function Install-Torch([string]$Exe, [string]$Flavor) {
    if ($Flavor -eq "rocm") {
        Run $Exe (@("-m", "pip", "install", "--no-cache-dir") + $RocmSdk) -FailMessage "AMD's ROCm SDK wheels did not install (docs\INSTALL.md)" | Out-Null
        Run $Exe @("-m", "pip", "install", "--no-cache-dir", $RocmTorch) -FailMessage "AMD's PyTorch wheel did not install" | Out-Null
    } elseif ($Flavor -eq "cuda") {
        Run $Exe @("-m", "pip", "install", "torch", "--index-url", $TorchCudaIndex) -FailMessage "PyTorch (CUDA) did not install" | Out-Null
    } else {
        Run $Exe @("-m", "pip", "install", "torch") -FailMessage "PyTorch did not install" | Out-Null
    }
}
function Set-Torch([string]$Exe, [string]$Want, [string]$Label) {
    $tv = Get-TorchVersion $Exe
    if (-not $tv) { Install-Torch $Exe $Want; return }
    $have = Get-TorchFlavor $tv
    if ($have -eq $Want) { Ok "torch $tv ($have build) in $Label"; return }
    $replace = [bool]$ReplaceTorch
    if (-not $replace) {
        if ($have -eq "rocm" -and $Want -eq "cpu") {
            Info "$Label has AMD's ROCm PyTorch ($tv). Its CPU math is much slower than the CPU build's (MMFF, dxtb,"
            Info "xtb); AIMNet2 on the GPU needs it. Suggested: the CPU build here, ROCm in .venv-rocm (-Gpu rocm; PERF-1)."
        }
        if (-not $DryRun -and (Test-Interactive)) {
            $answer = Read-Host "    Replace torch $tv ($have build) in $Label with the $Want build? [y/N]"
            $replace = $answer -match "^(y|yes)$"
        }
    }
    if ($replace) {
        Run $Exe @("-m", "pip", "uninstall", "-y", "torch") | Out-Null
        Install-Torch $Exe $Want
    } else {
        Ok "keeping torch $tv ($have build) in $Label; rerun with -ReplaceTorch to switch to the $Want build"
    }
}
Step "PyTorch in .venv"
$venvFlavor = "cpu"
if ($Gpu -eq "cuda") { $venvFlavor = "cuda" }
Set-Torch $VPy $venvFlavor ".venv"

# ------------------------------------------------------------------------------------- 4. QuantaBench
function Install-QuantaBench([string]$Dir) {
    $exe = Join-Path $Dir "Scripts\python.exe"
    $stamp = Join-Path $Dir ".quantabench-install"
    $want = "$(Get-Sha256 (Join-Path $Repo 'pyproject.toml')) extras=$Extras forks=$($ForkPkgs -join ',') $TbliteLinks"
    $here = $false
    if (Test-Path $exe) {
        $where = Get-Output $exe @("-c", $WhereCode)
        $src = (Join-Path $Repo "src") + "\"
        $here = [bool]($where -and $where.StartsWith($src, [StringComparison]::OrdinalIgnoreCase))
    }
    if (-not $Reinstall -and $here -and (Test-Path $stamp) -and ((Get-Content -Raw $stamp).Trim() -eq $want)) {
        Ok "installed from this checkout; pyproject.toml unchanged since the last install"
        return
    }
    $tv = Get-TorchVersion $exe
    $pipArgs = @("-m", "pip", "install", "-e", ".[$Extras]") + $ForkPkgs +
               @("--only-binary", "tblite", "--find-links", $TbliteLinks)
    if ($tv) {
        # keep the PyTorch build chosen above: an extra that wants another torch fails loudly instead
        $constraints = Join-Path $Dir ".torch-constraint.txt"
        Write-TextFile $constraints "torch==$tv"
        $pipArgs += @("--constraint", $constraints)
    }
    Push-Location $Repo
    try { Run $exe $pipArgs -FailMessage "pip install failed (output above); docs\INSTALL.md has fixes" | Out-Null }
    finally { Pop-Location }
    # PyPI's dxtb brought tad-dftd3 0.6 / tad-dftd4 0.8 (pinned to tad-mctc 0.7). The fork carries its own copies,
    # so a leftover install is unused and only makes `pip check` fail against tad-mctc 0.11.
    $stale = Get-Output $exe @("-c", $StaleTadCode)
    if ($stale) { Run $exe (@("-m", "pip", "uninstall", "-y") + ($stale -split " ")) -AllowFail | Out-Null }
    if (-not $DryRun) { [IO.File]::WriteAllText($stamp, $want + "`r`n", (New-Object Text.ASCIIEncoding)) }
}
Step "QuantaBench in .venv (pip install -e "".[$Extras]"")"
Install-QuantaBench $Venv

if ($Gpu -eq "rocm") {
    Step "AMD GPU environment (.venv-rocm)"
    New-Venv $VenvRocm
    $rocmPy = Join-Path $VenvRocm "Scripts\python.exe"
    if (-not $DryRun) {
        $rv = Get-PyVersion $rocmPy
        if ($rv -ne "3.12") { Fail ".venv-rocm uses Python $rv; AMD's wheels need 3.12. Remove .venv-rocm and rerun with -Python (a 3.12)." }
    }
    Set-Torch $rocmPy "rocm" ".venv-rocm"
    Install-QuantaBench $VenvRocm
    Info "check the GPU: .venv-rocm\Scripts\python.exe -c ""import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"""
}

# --------------------------------------------------------------------------------- 5. settings folder
if ($SettingsDir) {
    Step "Settings folder"
    $cli = Join-Path $Venv "Scripts\quantabench-cli.exe"
    $code = "import os, sys; from quantabench import settings; n = lambda p: os.path.normcase(os.path.realpath(os.path.expanduser(p))); " +
            "print('same' if n(sys.argv[1]) == n(str(settings.settings_dir())) else settings.settings_dir())"
    $current = $null
    if (Test-Path $VPy) { $current = Get-Output $VPy @("-c", $code, $SettingsDir) }
    if ($current -eq "same") {
        Ok "settings already in $SettingsDir"
    } else {
        if ($current) { Info "currently: $current" }
        if (-not (Test-Path $SettingsDir)) { Info "$SettingsDir does not exist yet: it is created (SyncThing can sync into it later)" }
        if ($env:QUANTABENCH_SETTINGS_DIR) { Warn "QUANTABENCH_SETTINGS_DIR is set and overrides this folder" }
        Show-Command $cli @("settings", "--set-dir", $SettingsDir)
        if (-not $DryRun) {
            $ErrorActionPreference = "Continue"
            & $cli settings --set-dir $SettingsDir | Out-Null
            if ($LASTEXITCODE -ne 0) { Fail "quantabench-cli settings --set-dir failed" }
            $ErrorActionPreference = "Stop"
            Info "settings are now read from $SettingsDir"
        }
    }
}

# ----------------------------------------------------------------------- 6. external programs (conda)
function Find-Conda {
    if ($Conda) {
        $c = Get-Command $Conda -ErrorAction SilentlyContinue
        if (-not $c) { Fail "-Conda $Conda not found" }
        return $c.Source
    }
    foreach ($name in @("conda", "mamba", "micromamba")) {   # conda first: it has `env update`
        $c = Get-Command $name -ErrorAction SilentlyContinue
        if ($c) { return $c.Source }
    }
    foreach ($p in @((Join-Path $MiniforgeDir "Scripts\conda.exe"),
                     (Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"),
                     (Join-Path $env:LOCALAPPDATA "miniforge3\Scripts\conda.exe"),
                     (Join-Path $env:USERPROFILE "mambaforge\Scripts\conda.exe"),
                     (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
                     (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
                     (Join-Path $env:USERPROFILE "micromamba\micromamba.exe"))) {
        if (Test-Path $p) { return $p }
    }
    return $null
}
function Get-EnvPrefix([string]$Tool, [string]$Name) {
    if (-not (Test-Path $Tool)) { return $null }
    $json = Get-Output $Tool @("env", "list", "--json")
    if (-not $json) { return $null }
    try { $envs = (ConvertFrom-Json $json).envs } catch { return $null }
    foreach ($p in $envs) { if ((Split-Path -Leaf $p) -eq $Name) { return $p } }
    return $null
}
function Set-CondaEnv([string]$Tool, [string]$File) {
    $name = $null
    foreach ($line in Get-Content $File) { if ($line -match "^name:\s*(\S+)") { $name = $Matches[1]; break } }
    if (-not $name) { Fail "$File has no name: line" }
    $micro = (Split-Path -Leaf $Tool) -like "micromamba*"
    $hash = Get-Sha256 $File
    $prefix = Get-EnvPrefix $Tool $name
    if ($prefix) {
        $stamp = Join-Path $prefix ".quantabench-env.sha256"
        if ((Test-Path $stamp) -and ((Get-Content -Raw $stamp).Trim() -eq $hash)) { Ok "$name is up to date with $(Split-Path -Leaf $File) ($prefix)"; return }
        Info "$name exists ($prefix): updating it from $(Split-Path -Leaf $File)"
        if ($micro) { Run $Tool @("install", "-y", "-n", $name, "-f", $File) | Out-Null }
        else { Run $Tool @("env", "update", "-n", $name, "-f", $File) | Out-Null }
    } else {
        if ($micro) { Run $Tool @("create", "-y", "-f", $File) | Out-Null }
        else { Run $Tool @("env", "create", "-f", $File) | Out-Null }
        $prefix = Get-EnvPrefix $Tool $name
    }
    if (-not $DryRun -and $prefix) { [IO.File]::WriteAllText((Join-Path $prefix ".quantabench-env.sha256"), $hash + "`r`n", (New-Object Text.ASCIIEncoding)) }
}
if ($WithEngines) {
    Step "External programs: xtb, MOPAC, Psi4 (conda environments)"
    $tool = Find-Conda
    if ($tool) {
        Ok "using $tool"
    } else {
        if ($MiniforgeDir -match " ") { Fail "Miniforge cannot be installed in a folder with spaces ($MiniforgeDir). Pass -MiniforgeDir C:\miniforge3" }
        if (Test-Path $MiniforgeDir) { Fail "$MiniforgeDir exists but has no Scripts\conda.exe; fix or remove it, or pass -Conda PATH" }
        Info "no conda, mamba or micromamba found: installing Miniforge into $MiniforgeDir (just for you, not on PATH)"
        $installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
        Show-Command "Invoke-WebRequest" @("-UseBasicParsing", "-Uri", $MiniforgeUrl, "-OutFile", $installer)
        if (-not $DryRun) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -UseBasicParsing -Uri $MiniforgeUrl -OutFile $installer
        }
        # Miniforge's documented silent install; /D= must come last and is never quoted
        $mfArgs = @("/S", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/D=$MiniforgeDir")
        Show-Command "Start-Process" @("-Wait", "-FilePath", $installer, "-ArgumentList", ($mfArgs -join " "))
        if (-not $DryRun) {
            $p = Start-Process -FilePath $installer -ArgumentList $mfArgs -Wait -PassThru
            if ($p.ExitCode -ne 0) { Fail "the Miniforge installer exited with code $($p.ExitCode)" }
            Remove-Item -Force $installer -ErrorAction SilentlyContinue
        }
        $tool = Join-Path $MiniforgeDir "Scripts\conda.exe"
        Info "Miniforge Prompt (Start Menu) gives you a terminal with conda"
    }
    Set-CondaEnv $tool (Join-Path $Repo "envs\tools-windows.yml")
    Set-CondaEnv $tool (Join-Path $Repo "envs\psi4.yml")
    Info "CREST has no Windows build: run it through WSL (docs\ENGINES.md). QuantaBench finds the rest by itself."
}

# ---------------------------------------------------------------------------------------- 7. shortcuts
function Set-Shortcut([string]$Path, [string]$Target, [string]$Description) {
    $shell = New-Object -ComObject WScript.Shell
    if (Test-Path $Path) {
        $lnk = $shell.CreateShortcut($Path)
        if ($lnk.TargetPath -ieq $Target -and $lnk.WorkingDirectory -ieq $Repo) { Ok "$Path"; return }
    }
    if ($DryRun) { Info "would create shortcut $Path -> $Target (start in $Repo)"; return }
    $lnk = $shell.CreateShortcut($Path)
    $lnk.TargetPath = $Target
    $lnk.WorkingDirectory = $Repo
    $lnk.Description = $Description
    $lnk.Save()
    Info "created $Path -> $Target"
}
if (-not $NoShortcut) {
    Step "Shortcuts (no console window)"
    # .venv\Scripts\quantabench.exe is pip's GUI launcher: it runs pythonw, so no console window opens
    $places = @([Environment]::GetFolderPath("Programs"))
    if ($DesktopShortcut) { $places += [Environment]::GetFolderPath("Desktop") }
    foreach ($place in $places) {
        Set-Shortcut (Join-Path $place "QuantaBench.lnk") (Join-Path $Venv "Scripts\quantabench.exe") "QuantaBench molecular modeling workbench"
        if ($Gpu -eq "rocm") {
            Set-Shortcut (Join-Path $place "QuantaBench (AMD GPU).lnk") (Join-Path $VenvRocm "Scripts\quantabench.exe") "QuantaBench with AMD's ROCm PyTorch"
        }
    }
    Info "QuantaBench.bat in the repository starts it the same way (QuantaBench.bat rocm: the GPU environment)"
}

# ------------------------------------------------------------------------------------------- 8. doctor
Step "quantabench-cli doctor"
$doctor = Run (Join-Path $Venv "Scripts\quantabench-cli.exe") @("doctor") -AllowFail
if ($doctor -ne 0) {
    Warn "doctor reported problems (above); docs\INSTALL.md has fixes"
    if ($sac -eq 1 -or $sac -eq 2) { Warn "'blocked by an Application Control policy' means Smart App Control: docs\INSTALL.md, 'Smart App Control'" }
}

Step "Done"
if (-not $NoShortcut) { Info "Start QuantaBench from the Start Menu (QuantaBench), or double-click QuantaBench.bat" }
Info "Command line: .venv\Scripts\quantabench-cli.exe (or activate: .\.venv\Scripts\Activate.ps1)"
Info "Update later: Help > Check for updates... in the app, quantabench-cli update, or run this script again."
