# QuantaBench (code)

Public repo `brandonbtm29/QuantaBench`: an open, GPU-aware molecular modeling workbench
(PySide6 GUI + CLI) that turns SMILES/names/lists into conformer ensembles refined with ML
potentials or xTB, with descriptors and exports for QSAR and LAMMPS.

Start with `docs/VISION.md` (why), `docs/ARCHITECTURE.md` (how), `docs/ROADMAP.md` and `docs/UI_PLAN.md` (what's next). Scheduled agent runs follow `docs/dev/RUNBOOK.md` and take work from `docs/dev/BACKLOG.md`.

## Layout rule (same as Brandon's other repos)
- **Code** lives here: `GitHub Projects/QuantaBench` (Windows `C:\Users\Brand\GitHub Projects\QuantaBench`).
  It syncs between machines through GitHub only.
- **Settings** live in `~/SyncThing/QuantaBench/Settings/settings.json`. A git-ignored
  `settings_location.json` in the repo root points there (`quantabench-cli settings --set-dir ...`).
- **Results and data** go to the output folder in settings (Brandon's: `~/SyncThing/QuantaBench/Runs`).
  Never commit data, results, model weights or settings.

## Conventions
- Python 3.12 (AMD's Windows ROCm PyTorch wheels are 3.12-only); `requires-python >= 3.11`.
- PySide6 (not PyQt6): LGPL, and OVITO's embeddable viewport is PySide6-only.
- PyTorch is the GPU layer (ROCm/CUDA/MPS all appear as torch devices). Import torch, aimnet,
  dxtb and ovito lazily; the core install must work without them (CI runs without torch).
- The pipeline (`chem/pipeline.py`) is UI-agnostic; GUI and CLI only call it. Long work runs on
  `QThread` workers (`gui/workers.py`).
- Target Windows 11 native, WSL2 and Linux; macOS later. Use `pathlib`, no hardcoded separators.
- Lint with `ruff check .`, test with `pytest` (`xvfb-run -a pytest` on headless Linux).
- Commits are SSH-signed (configured globally). Don't disable signing.
- Don't start the GUI or long batch runs in the background without saying so.
- Keep scratch scripts in `scratch/` (gitignored).

## Hardware notes (Brandon's workstation BTM NOVA)
- Ryzen 9 7900X (12C/24T), Radeon RX 9060 XT 16 GB (gfx1200), Windows 11.
- WSL2 GPU pass-through for this card was fragile in earlier testing (`Ioctl failed: -22`); prefer
  native Windows for GPU work. LAMMPS GPU via OpenCL on native Windows gave ~9x over 12-core CPU.
