@echo off
rem Launch QuantaBench from this repo's .venv (create it first, see docs\INSTALL.md)
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m quantabench
) else (
    echo No .venv found. See docs\INSTALL.md
    pause
)
