# Installing QuantaBench

## Quick install

One script per OS does the whole installation. It is safe to run again: every step that is already
done is skipped, so rerunning it after `git pull` is also how you repair or finish an install. Add
`--dry-run` (Windows: `-DryRun`) to see every step without changing anything.

You need **git** and **Python 3.12**. The scripts look for Python 3.12 and say how to get it if it is
missing; they never install system Python themselves:

- macOS: `brew install python@3.12`, or the python.org installer. The first `git` command offers to install
  Apple's command line tools.
- Windows: `winget install Python.Python.3.12` and `winget install Git.Git`.
- Linux: `sudo apt install python3.12 python3.12-venv git` (Ubuntu 24.04).

### macOS (Apple Silicon)

In Terminal:

```bash
mkdir -p ~/GitHub\ Projects && cd ~/GitHub\ Projects && git clone https://github.com/brandonbtm29/QuantaBench.git && cd QuantaBench && bash scripts/install_macos.sh --with-engines --settings-dir ~/SyncThing/QuantaBench/Settings
```

Run it again later (repair, or after an update):

```bash
cd ~/GitHub\ Projects/QuantaBench && git pull --ff-only && bash scripts/install_macos.sh --with-engines --settings-dir ~/SyncThing/QuantaBench/Settings
```

Start QuantaBench from **~/Applications/QuantaBench.app** (Launchpad, Spotlight; drag it to the Dock),
or double-click `QuantaBench.command` in the repository folder.

### Windows 11 (BTM NOVA)

In PowerShell, for the existing checkout:

```powershell
cd "$HOME\GitHub Projects\QuantaBench"; git pull; powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -WithEngines -SettingsDir "$HOME\SyncThing\QuantaBench\Settings"
```

On a PC without a checkout yet:

```powershell
mkdir "$HOME\GitHub Projects" -Force | Out-Null; cd "$HOME\GitHub Projects"; git clone https://github.com/brandonbtm29/QuantaBench.git; cd QuantaBench; powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -WithEngines -SettingsDir "$HOME\SyncThing\QuantaBench\Settings"
```

Start QuantaBench from the **Start Menu** ("QuantaBench"), or double-click `QuantaBench.bat`. Neither
opens a console window.

On BTM NOVA, `.venv` currently holds AMD's ROCm PyTorch. The script asks before it replaces it. Answering
`y` installs PyPI's CPU build, which is the faster one for everything QuantaBench runs today (PERF-1).
Add `-Gpu rocm` to keep a separate GPU environment, `.venv-rocm`, with AMD's wheels and its own Start Menu
entry, "QuantaBench (AMD GPU)".

### Linux and WSL2

```bash
mkdir -p ~/GitHub\ Projects && cd ~/GitHub\ Projects && git clone https://github.com/brandonbtm29/QuantaBench.git && cd QuantaBench && bash scripts/install_linux.sh --with-engines --settings-dir ~/SyncThing/QuantaBench/Settings
```

Start QuantaBench from the application menu (in WSL, WSLg also adds it to the Windows Start Menu). In
WSL, clone into the Linux home as above, not into `/mnt/c/...`. The script refuses to put a Linux venv on
top of the Windows checkout's `.venv`.

### What the scripts do

1. **Python 3.12.** The scripts find it (`py -3.12` on Windows) or explain how to get it.
2. **`.venv`.** They create it in the repository folder, or reuse it.
3. **PyTorch first**, so pip cannot pull a build that does not match your GPU:

   | | default | other choices (`--gpu` / `-Gpu`) |
   |---|---|---|
   | Windows | `cpu`: PyPI's build, the CPU build on Windows | `rocm`: also `.venv-rocm` with AMD's ROCm 7.2.1 wheels; `cuda`: NVIDIA |
   | Linux, WSL2 | `cpu`: download.pytorch.org's CPU wheels | `cuda`: PyPI's build; `rocm`: download.pytorch.org's ROCm wheels; `auto`; `--torch-index URL` |
   | macOS | PyPI's build, which runs on the CPU and on the Apple GPU (`mps`) | |

   An installed PyTorch of another build is never replaced without asking. Pass `--replace-torch`
   (`-ReplaceTorch`) to replace it without the question.
4. **`pip install -e ".[all,dev]"`**, with PyTorch pinned to the installed build (Linux adds
   `tad-libcint<0.4` for GFN2-xTB in dxtb; macOS never builds tblite from source). The scripts skip this
   step when `pyproject.toml` has not changed since the last run (`--reinstall` forces it).
5. **`--settings-dir DIR`** runs `quantabench-cli settings --set-dir DIR`
   ([Settings in a synced folder](#settings-in-a-synced-folder)).
6. **`--with-engines`** installs Miniforge into `~/miniforge3` (Windows: `%USERPROFILE%\miniforge3`, just for
   you, not on PATH) when no conda, mamba or micromamba is found. Then it creates or updates the conda
   environments: `envs/tools.yml` (xtb, CREST, MOPAC), or `envs/tools-windows.yml` (xtb, MOPAC) on Windows,
   plus `envs/psi4.yml`. Each environment is updated only when its file changed. `--conda PATH` picks the
   tool.
7. **A launcher that hides the terminal window**:
   - Windows: Start Menu shortcut to `.venv\Scripts\quantabench.exe`, pip's GUI launcher, which runs
     `pythonw`. `-DesktopShortcut` adds one on the Desktop.
   - macOS: `~/Applications/QuantaBench.app` (`--no-app` skips it), plus `QuantaBench.command`.
   - Linux: `~/.local/share/applications/quantabench.desktop`.
8. **`quantabench-cli doctor`** at the end, which shows what QuantaBench can use.

Windows also checks for **Smart App Control**, which blocks new DLLs ([below](#smart-app-control)), and
warns about **spaces in paths**. Spaces are fine for QuantaBench. They only produce harmless noise from
AMD's ROCm PyTorch, and Miniforge must not be installed in a folder with spaces.

## Updating

- **In the app:** Help > Check for updates... It shows the new commits and offers **Update and restart**.
- **On the command line:** `quantabench-cli update --check` reports how far the checkout is behind
  `origin/main`, and `quantabench-cli update` updates it.

Both run `git pull --ff-only`. If `pyproject.toml`'s dependencies or entry points changed, they then
reinstall with `pip install -e`, using the extras that are already installed and keeping the installed
PyTorch. They refuse, and say why, in three cases:

- tracked files have local changes (commit or `git stash` them first)
- the checkout is on another branch
- it has commits of its own that `origin/main` does not have

An install that is not a git checkout gets instructions instead. Running the install script again also
updates, except for the `git pull`.

## Manual installation

The sections below are the steps the scripts automate, with the details per platform.

QuantaBench needs Python 3.11 or newer. **Use Python 3.12** if you want AMD GPU support on Windows,
because AMD's Windows PyTorch wheels are built for 3.12 only.

The order matters: **install the PyTorch build that matches your GPU first**, then QuantaBench.
Otherwise pip may pull a generic PyTorch that cannot see your card.

## Windows 11, AMD Radeon (native, recommended for RX 7000/9000)

Native Windows is the fastest and most reliable route for Radeon cards. Earlier testing on this project
found WSL2 GPU pass-through fragile on consumer Radeons. `scripts\install_windows.ps1 -Gpu rocm` does
steps 2 to 4 in a second environment, `.venv-rocm`, and keeps the CPU build in `.venv`
([below](#what-the-amd-gpu-can-do-today-windows-rocm-721)).

1. Install AMD Software: Adrenalin Edition 26.2.2 or newer, and Python 3.12 from python.org.
2. Create an environment:

   ```powershell
   cd "$HOME\GitHub Projects\QuantaBench"
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

3. Install AMD's ROCm SDK and PyTorch wheels (ROCm 7.2.1; check
   [AMD's page](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
   for newer releases):

   ```powershell
   pip install --no-cache-dir `
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
   pip install --no-cache-dir `
     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   On ROCm, PyTorch still calls the GPU `cuda`. `True` plus your Radeon's name means it works.

4. Install QuantaBench with the optional engines:

   ```powershell
   pip install -e ".[all,dev]"
   quantabench-cli doctor --bench
   quantabench
   ```

   If `doctor` reports a PyTorch version ending in `+cpu` (for example `2.14.1+cpu`), pip pulled the
   generic CPU build from PyPI because step 3 was skipped or ran afterwards. Rerun step 3 with
   `--force-reinstall --no-deps` on the torch wheel. The ML packages accept torch 2.9.

### Smart App Control

Windows 11's Smart App Control blocks unsigned DLLs that Microsoft's cloud has not seen before.
Brand-new releases of scientific wheels (RDKit, VTK, PyTorch extensions) often fall into that group.
The symptom is:

```
ImportError: DLL load failed while importing rdForceField: An Application Control policy has blocked this file.
```

There are two ways around it:

- **Turn Smart App Control off.** Windows Security > App & browser control > Smart App Control
  settings > Off. Since the April 2026 Windows update it can be turned back on later without
  reinstalling Windows. This is the practical choice for a development machine that runs many
  pip-installed native libraries.
- **Keep it on and use older releases** that Microsoft already recognizes, for example
  `pip install "rdkit==2026.3.6"`. This is not guaranteed, and other packages may be blocked next.

Organization-managed PCs (WDAC/App Control for Business) show the same message. There, ask IT to
allow the Python environment, or use WSL2/Linux.

### What the AMD GPU can do today (Windows, ROCm 7.2.1)

PyTorch sees the card, but no QuantaBench engine uses it yet: AMD's Windows PyTorch lacks GPU linear
algebra (MAGMA), which dxtb needs, and AIMNet2's dispersion runs through NVIDIA-only Warp kernels
(backlog GPU-1/GPU-2). AMD's wheel also has a much slower CPU math library than PyPI's CPU build. Until
that changes, keep the CPU build in `.venv` for everyday work and install ROCm into a second
environment (`py -3.12 -m venv .venv-rocm`) for GPU experiments.

### What native Windows can't do yet

- **GFN2-xTB, xTB polarizability and xTB IR intensities** need dxtb's `libcint` integral backend.
  The only `tad-libcint` release with a Windows wheel (0.4) is incompatible with dxtb 0.4. On Windows,
  GFN1-xTB optimizations, frequencies, thermochemistry, Mulliken charges and charge-based dipoles all
  work; polarizability and IR intensities are skipped with a note. **GFN2-xTB itself is available on
  Windows through the xtb program** (`xtb-gfn2`, with ALPB solvation, its own optimizer, dipoles,
  D4 polarizability and finite-difference IR intensities): install `envs/tools-windows.yml`. Tested on
  Linux so far; the first Windows run is backlog ENG-1.
- **pyvistaqt is pinned below 0.13.** 0.13 replaced the native-window VTK widget with a framebuffer
  `QOpenGLWidget` that crashes in `initializeGL` with the AMD driver on Windows (verified on an
  RX 9060 XT, driver 26.10.1). 0.12 works. If you upgraded it by hand: `pip install "pyvistaqt==0.12.0"`.

- **LAMMPS SMD runs are intermittently unstable with the Windows wheel.** Identical SMD inputs fail some
  runs with "Out of range atoms - cannot compute PPPM": ~5-10% with a free Na⁺, and 2 of 6 for a TFA
  anion on graphite without a counter-ion (2026-11-01). Linux/WSL runs are deterministic. Until MD-1 is
  solved, run production SMD with the LAMMPS Windows installer's `lmp.exe` or in WSL.

## Windows + WSL2 (Ubuntu 22.04 / 24.04)

AMD lists the RX 9060 XT as supported on WSL2 with ROCm 7.2.1 and the *Adrenalin 26.1.1 for WSL2*
driver. Follow AMD's "Radeon on WSL" guide, then install the Linux ROCm PyTorch wheels inside WSL as
in the Linux section. If `torch.cuda.is_available()` is `False` or WSL logs `Ioctl failed: -22`, use
native Windows for GPU work. WSL2 is still useful on the CPU, for example to run GFN2-xTB, whose
libcint backend has no Windows build yet:

```bash
pip install "tad-libcint<0.4"   # enables gfn2-xtb in dxtb on Linux/WSL
```

## Linux (native)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# Install PyTorch first, using the command from https://pytorch.org/get-started/locally/
# for your GPU (ROCm for AMD, CUDA for NVIDIA, or CPU). For example, CPU only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev]" "tad-libcint<0.4"
quantabench-cli doctor --bench
```

The GUI needs `libxcb-cursor0` on Debian/Ubuntu (`sudo apt install libxcb-cursor0`).

## macOS, Apple Silicon (being brought up: PLAT-1)

Not tested on a Mac yet. The steps below are expected to work; report what doesn't. See
[PLATFORMS.md](PLATFORMS.md) for the platform matrix.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                 # PyPI's macOS build includes the Apple GPU backend (mps)
pip install -e ".[all,dev]"       # tblite's wheel needs macOS 15 or newer
quantabench-cli doctor --bench
conda env create -f envs/tools.yml    # xtb, CREST and MOPAC have native osx-arm64 builds
conda env create -f envs/psi4.yml
```

What to expect on a Mac today:
- The Apple GPU (`mps`) has no float64. GPU engines run in float32.
- AIMNet2 and dxtb need FORK-3 and FORK-4 before they run on the Mac GPU. Until then they use the CPU.
- dxtb's GFN2 (libcint) is not available yet (FORK-1). Use `tblite-gfn2` or `xtb-gfn2` for GFN2.
- CREST runs natively, unlike on Windows.

## External programs: xtb, CREST, MOPAC, Psi4 (optional)

These are separate programs, not pip packages. Install them with conda (Miniforge) into their own
environments; QuantaBench finds them there by itself (`quantabench-cli doctor` lists them). The install
scripts' `--with-engines` (`-WithEngines`) does this, Miniforge included. By hand:

```bash
conda env create -f envs/tools.yml          # Linux/WSL: xtb, CREST, MOPAC
conda env create -f envs/tools-windows.yml  # native Windows: xtb, MOPAC (CREST runs through WSL)
conda env create -f envs/psi4.yml           # Psi4 for the QM (DFT) stage, with the pins it needs
```

Set paths by hand in Settings > Engines when they live elsewhere. Details, versions and pitfalls per
program are in [ENGINES.md](ENGINES.md).

## Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `ml` | `aimnet` | AIMNet2 ML potential (downloads model weights on first use) |
| `xtb` | `dxtb`, `tblite` (not on Windows) | GFN1-xTB (dxtb), GFN2-xTB in process (tblite; dxtb with `tad-libcint`) |
| `interop` | `cclib` | Import Gaussian/ORCA/Psi4/xtb/MOPAC output files |
| `protonate` | `dimorphite-dl` | Alternative pH protonation |
| `ovito` | `ovito` | Embedded OVITO viewport |
| `md` | `lammps` | LAMMPS Python module: `md smd --run` and quick checks of SMD setups |
| `excel` | `openpyxl` | `.xlsx`/`.xlsm` input (CompTox dashboard exports) |
| `all` | ml, xtb, protonate, md, excel, interop | |

Notes on the new extras:

- **`md`.** PyPI has `lammps` wheels for Windows (win_amd64), Linux and macOS. The wheel is a CPU
  build with OpenMP (the Linux wheel tested here has no GPU package). For production pulls, and for
  the GPU package (OpenCL on Radeon), use the LAMMPS Windows installer's `lmp.exe`; every SMD folder's
  `README.txt` lists the exact commands, e.g. `lmp.exe -sf gpu -pk gpu 1 neigh no -in in.smd`.
  QuantaBench only writes the input files and parses the outputs, so the extra is optional.
- **`excel`.** openpyxl is pure Python. Legacy `.xls` files additionally need `xlrd`; otherwise save
  them as `.xlsx` or `.csv`.

## Settings in a synced folder

```powershell
quantabench-cli settings --set-dir "~/SyncThing/QuantaBench/Settings"
```

This writes a git-ignored `settings_location.json` pointer in the repo, so every computer that clones
the repo and has the SyncThing folder shares the same settings. The GUI offers the same choice under
Settings > Folders, and the install scripts under `--settings-dir` (`-SettingsDir`).
