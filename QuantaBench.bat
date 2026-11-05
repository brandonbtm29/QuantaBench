@echo off
rem Start QuantaBench without a console window: pythonw from this repo's .venv.
rem   QuantaBench.bat        the normal environment (.venv)
rem   QuantaBench.bat rocm   the AMD GPU environment (.venv-rocm, made by install_windows.ps1 -Gpu rocm)
rem Install or repair: powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
cd /d "%~dp0"
set "QB_VENV=.venv"
if /i "%~1"=="rocm" set "QB_VENV=.venv-rocm"
if exist "%QB_VENV%\Scripts\pythonw.exe" (
    start "" "%QB_VENV%\Scripts\pythonw.exe" -m quantabench
) else (
    echo No %QB_VENV% found in %CD%. Install QuantaBench first:
    echo     powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
    pause
)
