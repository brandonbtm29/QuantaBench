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
`y` installs PyPI's CPU build, which is much faster for CPU work (MMFF, dxtb, xtb). Add `-Gpu rocm` to keep a
separate GPU environment, `.venv-rocm`, with AMD's wheels and its own Start Menu entry, "QuantaBench (AMD
GPU)": AIMNet2 runs about 10x faster there on the RX 9060 XT. Which one should be the default is backlog
PERF-1. Answering `n` keeps the ROCm build in `.venv`.

The Windows and macOS scripts also install the two packages PyPI cannot provide there: the dxtb fork with
tad-libcint 0.4 (GFN2-xTB, polarizability and IR intensities in dxtb) and, on Windows, tblite from the
fork's wheels (details below and in [PLATFORMS.md](PLATFORMS.md#4-forks-builds-and-upstream-fixes)). The
pins are in [`scripts/fork-packages.json`](#fork-packages-scriptsfork-packagesjson), so a new one also
arrives through Help > Check for updates.

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
4. **`pip install -e ".[all,dev]"`**, with PyTorch pinned to the installed build, plus this platform's
   packages from `scripts/fork-packages.json` (Windows: the dxtb fork, tad-libcint 0.4 and the fork's tblite
   wheels; macOS: the dxtb fork and tad-libcint 0.4, and tblite never built from source; Linux:
   `tad-libcint<0.4` for GFN2-xTB in dxtb). The scripts skip this
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
  Tick **Also upgrade the installed packages** to upgrade the packages too. With nothing to pull, the same
  window offers **Upgrade packages...**.
- **On the command line:** `quantabench-cli update --check` reports how far the checkout is behind
  `origin/main`, and `quantabench-cli update` updates it. `quantabench-cli update --upgrade-packages` also
  upgrades the packages, with or without new commits.

Both run `git pull --ff-only`. Then they reinstall with `pip install -e` when one of these changed:

- `pyproject.toml`'s dependencies or entry points
- this platform's entry in `scripts/fork-packages.json` (below)

The reinstall uses the extras that are already installed, adds the fork packages and keeps the installed
PyTorch.

They refuse, and say why, in three cases:

- tracked files have local changes (commit or `git stash` them first)
- the checkout is on another branch
- it has commits of its own that `origin/main` does not have

An install that is not a git checkout gets instructions instead. Running the install script again also
updates, except for the `git pull`.

**What a plain update does not do.** Packages that are already installed and still satisfy
`pyproject.toml` stay at their version, because pip only installs what is missing. Upgrading them is the
opt-in `--upgrade-packages` (the check box in the app). It runs:

```
pip install -e ".[<installed extras>]" <fork packages> --upgrade --upgrade-strategy eager --constraint <torch==installed>
```

- `eager` upgrades every dependency to the newest version `pyproject.toml` allows.
  `--upgrade-strategy only-if-needed` upgrades only QuantaBench and the fork packages, and dependencies only
  where they must move.
- **PyTorch** stays the installed build (ROCm, CUDA or CPU), pinned by a constraints file.
- **The fork packages** stay. They are named on pip's command line, so pip never swaps the dxtb fork for
  PyPI's dxtb. Those from a git tag or a release page are then reinstalled from it
  (`--force-reinstall --no-deps`).
- **The result lists what changed** (`pip list` before and after): `numpy 2.1.0 -> 2.3.4`, new packages and
  removed ones. The app shows the list and asks before it restarts.
- **On Windows, close other QuantaBench windows and any `quantabench-cli worker` first.** Windows keeps the
  files of a running program locked, so pip may fail to replace a package that is in use. If it does, the
  message says so: close the app and run `quantabench-cli update --upgrade-packages` in a terminal. pip can
  leave folders named `~...` in `.venv\Lib\site-packages` with the old files of a package that was in use;
  they can be deleted after a restart.
- On macOS, tblite's `libomp.dylib` is linked to PyTorch's again after pip, as the installer does
  ([below](#macos-apple-silicon-being-brought-up-plat-1)).

**Workers.** A running `quantabench-cli worker` keeps the code it started with. Restart it after an update
([WORKERS.md](WORKERS.md#commands)).

### Fork packages: `scripts/fork-packages.json`

pip cannot get every package from PyPI on every platform, and `pyproject.toml` cannot name a git tag or a
release page (PyPI refuses packages that depend on direct URLs). The installers therefore add those
packages, and the list lives in one file that all three installers and `quantabench-cli update` read:

```json
"windows": {
  "packages": ["dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1", "tad-libcint>=0.4", "tblite"],
  "only_binary": ["tblite"],
  "find_links": {"tblite": "https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1"}
}
```

Each platform entry (`windows`, `macos`, `linux`) has the same three keys:

- `packages`: requirement strings added to `pip install -e`.
- `only_binary`: names passed as `--only-binary`, so pip never builds them from source.
- `find_links`: package name -> URL, passed as `--find-links`.

Change an entry, and the next update (in the app or `quantabench-cli update`) reinstalls with it. For a
package from a git tag or a release page, it runs `pip install --force-reinstall --no-deps` for that package,
because a new build often keeps its version number. That applies when the package's requirement or
`find_links` URL changed. pip handles a changed version range by itself. Keep the file ASCII: Windows
PowerShell 5.1 reads it as ANSI. The install hints in `quantabench-cli doctor`
(`chem/engines.py` `DXTB_FORK_REQ`, `chem/eng_tblite.py` `WINDOWS_WHEELS`) name the same pins, and
`tests/test_install_scripts.py` fails when they drift apart.

The installers' pip run does not replace an installed package whose version still fits, so a new tblite
build under the same version number reaches them only after `pip uninstall tblite`. Give a new build a new
version (e.g. `0.8.0.post1`) and pin it (`tblite==0.8.0.post1`) so the installers pick it up too. Update
reinstalls it either way.

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
   # dxtb GFN2-xTB, polarizability and IR intensities (libcint backend, see below):
   pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"
   # tblite: in-process GFN2/GFN1-xTB with ALPB/GBSA/CPCM (Windows wheels from the fork):
   pip install tblite --only-binary tblite --find-links https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1
   quantabench-cli doctor --bench
   quantabench
   ```

   The tblite line adds `tblite-gfn2`/`tblite-gfn1` (in-process GFN2/GFN1-xTB with ALPB, GBSA
   and CPCM). PyPI has no Windows wheel for it, so the `xtb` extra skips it on Windows; these wheels
   (CPython 3.11-3.13, win_amd64) are built from tblite 0.8.0 in Brandon's fork (FORK-2; one build-file
   fix, no source changes), with the MinGW-w64 runtime and reference BLAS/LAPACK bundled.
   `--only-binary tblite` stops pip from picking a newer PyPI sdist that it cannot compile.

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

- **AIMNet2 runs on the GPU** (measured on an RX 9060 XT: about 10x faster than the 12-core CPU for a
  batch of PFOA conformers). aimnet's own calculator needs NVIDIA Warp, which only supports CUDA, so
  QuantaBench switches to its Warp-free path (`chem/aimnet_torch.py`, dispersion from dxtb's `tad-dftd3`) on
  ROCm and Apple GPUs. `quantabench-cli doctor` shows which path each device gets;
  `QUANTABENCH_AIMNET_BACKEND=stock|torch` forces one.
- **dxtb (GFN1/GFN2-xTB) stays on the CPU by default.** AMD's Windows PyTorch has no GPU Cholesky (no
  MAGMA); QuantaBench can run those steps on the CPU inside a GPU calculation, and the results match,
  but dxtb 0.4 measured 5-8x slower on this GPU than on the CPU. Settings > Compute > "dxtb on the
  GPU" (or `QUANTABENCH_DXTB_GPU=on`) uses the GPU anyway.
- A process that used the GPU can hang at exit in AMD's runtime (it spins one core after finishing).
  QuantaBench's own entry points (`quantabench`, `quantabench-cli`, lane processes, the test suite) exit
  at once with `os._exit` when they touched a ROCm GPU on Windows; after `git pull`, run
  `pip install -e .` once so the launchers pick this up. `QUANTABENCH_HARD_EXIT=0` turns it off.
- AMD's wheel has a much slower CPU math library than PyPI's CPU build (52 vs 881 GFLOP/s matmul on
  NOVA). So keep the CPU build in `.venv` for CPU work (MMFF, dxtb, xtb) and the ROCm build in
  `.venv-rocm` for AIMNet2 on the GPU: `scripts\install_windows.ps1 -Gpu rocm` sets up both (backlog
  PERF-1 tracks which should be the default).

### What native Windows can't do yet

- **dxtb's GFN2-xTB, xTB polarizability and xTB IR intensities need the dxtb fork** on Windows. They
  use dxtb's `libcint` integral backend, and the only `tad-libcint` release with a Windows wheel (0.4)
  does not work with PyPI's dxtb 0.4; install the fork as in step 4 (details in
  [dxtb's libcint backend](#dxtbs-libcint-backend-gfn2-polarizability-ir-intensities)). Without it,
  GFN1-xTB optimizations, frequencies, thermochemistry, Mulliken charges and charge-based dipoles all
  work; polarizability and IR intensities are skipped with a note, and `quantabench-cli doctor` says
  what is missing. **GFN2-xTB itself is available on Windows through the xtb program** (`xtb-gfn2`,
  with ALPB solvation, its own optimizer, dipoles, D4 polarizability and finite-difference IR
  intensities): install `envs/tools-windows.yml`. Tested on Linux so far; the first Windows run is
  backlog ENG-1. It also runs in process through tblite (`tblite-gfn2`, the third pip line of step 4),
  with dipoles, a finite-field polarizability and ALPB/GBSA/CPCM solvation.
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
native Windows for GPU work. WSL2 is still useful on the CPU, for example for CREST. dxtb's GFN2
works there with PyPI's dxtb, as on Linux:

```bash
pip install "tad-libcint<0.4"   # enables gfn2-xtb in PyPI's dxtb on Linux/WSL (or the fork, see below)
```

## Linux (native)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# Install PyTorch first, using the command from https://pytorch.org/get-started/locally/
# for your GPU (ROCm for AMD, CUDA for NVIDIA, or CPU). For example, CPU only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev]" "tad-libcint<0.4"
# or, the same stack as on Windows/macOS (dxtb fork, tad-libcint 0.4):
#   pip install -e ".[all,dev]" "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"
quantabench-cli doctor --bench
```

The GUI needs `libxcb-cursor0` on Debian/Ubuntu (`sudo apt install libxcb-cursor0`).

## macOS, Apple Silicon (being brought up: PLAT-1)

`scripts/install_macos.sh --with-engines` (top of this file) does all of this; the manual steps are below. See
[PLATFORMS.md](PLATFORMS.md) for the platform matrix and the first full test run on the MacBook (2026-11-06).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                 # PyPI's macOS build includes the Apple GPU backend (mps)
pip install -e ".[all,dev]"       # tblite's wheel needs macOS 15 or newer
pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"   # dxtb GFN2, IR
quantabench-cli doctor --bench
conda env create -f envs/tools.yml    # xtb, MOPAC (and conda-forge's CREST, which fails on macOS: see below)
conda env create -f envs/psi4.yml
```

What to expect on a Mac (first run on Brandon's MacBook Pro, Apple Silicon, macOS 27, 2026-11-05):
- The Apple GPU (`mps`) has no float64. AIMNet2 runs there in float32 through QuantaBench's Warp-free path
  (energies match the CPU to ~1e-5 eV), but for PFAS-sized molecules the M-series CPU is faster
  (benchmark: 8,230 vs 5,305 molecules/h), so `lanes auto` gives CPU lanes when the benchmark says so.
  Refining 6 PFAS (42 conformers) in one process on the M3 Max: 16.0 s on the CPU (the default, one conformer
  at a time), 9.9 s batched on `mps`, 8.7 s batched on the CPU (`--batched on`).
- dxtb (GFN1/GFN2) stays on the CPU on MPS by design (no float64, incomplete linear algebra). Its GFN2,
  polarizability and IR intensities work with the dxtb fork and tad-libcint 0.4 (the installer brings both).
- `tblite-gfn2` (gas and ALPB water) runs in process: 2,860 / 2,547 molecules/h on the benchmark set. The
  installer links tblite's OpenMP runtime to PyTorch's (see below).
- xtb, MOPAC and Psi4 run natively from the conda environments (`--with-engines`). QuantaBench finds
  them in `~/miniforge3/envs`, Homebrew's Miniforge cask and any environment conda lists in
  `~/.conda/environments.txt`, without a PATH change.
- CREST: conda-forge's osx-arm64 build (in `qb-tools`) links Apple's Accelerate and fails every optimization of
  19 atoms or more; QuantaBench flags each such molecule and `doctor` names the build. Use the fork's macOS
  build and set it as this Mac's path: `quantabench-cli settings --engine-path crest=<folder>/crest-macos-arm64/bin/crest`
  (docs/ENGINES.md, CREST; the installer does it once the build is released, FORK-5).
- LAMMPS (`md`): PyPI's macOS wheel links MPICH's `libmpi.12.dylib` without shipping it, so `import lammps`
  works and running LAMMPS fails with `Library not loaded: @rpath/libmpi.12.dylib`. The `md` extra installs
  `lammps[mpi]` on macOS, which brings PyPI's `mpich` wheel; an older install needs `pip install mpich` (or a
  rerun of the installer).
- MOPAC 23.2.5's PM7 polarizability differs from the Linux build's (water 1.347 vs 1.259 Å³, the same on
  Windows); the heat of formation, dipole and orbital energies agree. Don't mix MOPAC polarizabilities from
  Linux and macOS/Windows in one table (WIN-2 in the backlog).
- A worker (`quantabench-cli worker`) should be started from Terminal in your own session (or the LaunchAgent
  in docs/WORKERS.md), not over SSH: a worker started over SSH listened on its Tailscale address but received
  no connections.

**tblite and PyTorch on macOS.** tblite's wheel and PyTorch each bundle LLVM's OpenMP runtime (`libomp.dylib`),
and a process that loads both aborts with `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized`. The installer links tblite's copy to PyTorch's (`tblite/.dylibs/libomp.dylib` becomes a
symlink; the original is kept as `libomp.dylib.qb-orig`), so one runtime loads. After a `pip install` of tblite by
hand, rerun the installer; `quantabench-cli doctor` lists tblite as unavailable, with the fix, until then.

## dxtb's libcint backend (GFN2, polarizability, IR intensities)

dxtb's GFN2-xTB (`gfn2-xtb`) and every dxtb electric-field property (xTB polarizability, IR
intensities from dipole derivatives) use dipole and quadrupole integrals from `tad-libcint`. Two
working stacks exist (FORK-1 in [PLATFORMS.md](PLATFORMS.md)):

| | PyPI dxtb 0.4 | dxtb fork 0.4.0.post1 |
|---|---|---|
| Install | `pip install "tad-libcint<0.4"` | `pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"` |
| Linux, WSL | yes | yes |
| Windows | no (tad-libcint 0.3 has no Windows wheel) | yes |
| macOS arm64 | no | yes (fork CI; not tried on a Mac yet) |
| Pulls in | tad-mctc 0.7 | tad-mctc 0.11, tad-multicharge 0.8 (tad-dftd3/-dftd4 are vendored in the fork) |

The fork is dxtb 0.4.0 ported to the new tad-mctc API. Its results match PyPI's stack to about 1e-8
(relative): tad-mctc 0.9 moved to CODATA 2018 constants, which shifts GFN2 energies by ~1e-6 eV for
water; frequencies and IR intensities agree to better than 1e-5 cm⁻¹ and 1e-5 km/mol. It installs
as `dxtb 0.4.0.post1`, so `dxtb>=0.4` (the `xtb` extra) stays satisfied and later installs keep it;
add it to the same `pip install -e ".[all,dev]" ...` command or run it afterwards. pip refuses to
combine `tad-libcint>=0.4` with PyPI's dxtb (conflicting tad-mctc pins). If both got installed anyway
(e.g. with `--no-deps`), `quantabench-cli doctor` reports why GFN2 is unavailable and prints the
install line above.

AIMNet2's Warp-free GPU path uses the same `tad-dftd3` as the installed dxtb: the fork's vendored copy
with the fork, PyPI's `tad-dftd3` 0.6 with PyPI's dxtb. Don't add `tad-dftd3` to a fork install: every
tad-dftd3 release pins tad-mctc exactly (0.6 → 0.7, 0.7 → 0.8), so pip reports `ResolutionImpossible`
against the fork's tad-mctc 0.11. A `tad-dftd3`/`tad-dftd4` left over from an earlier PyPI install is
harmless (QuantaBench uses the fork's copy first); the Windows and macOS installers remove it, or run
`pip uninstall tad-dftd3 tad-dftd4`.

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
| `ml` | `aimnet` | AIMNet2 ML potential (downloads model weights on first use). Its GPU path on AMD and Apple GPUs takes D3 dispersion from the `tad-dftd3` that comes with dxtb (`xtb` extra or the fork), or `pip install tad-dftd3` alone with PyPI's dxtb stack |
| `xtb` | `dxtb`, `tblite` (Windows: separate line, step 4 above) | GFN1-xTB (dxtb), GFN2-xTB in process (tblite; dxtb with `tad-libcint`, see [above](#dxtbs-libcint-backend-gfn2-polarizability-ir-intensities)) |
| `interop` | `cclib` | Import Gaussian/ORCA/Psi4/xtb/MOPAC output files |
| `protonate` | `dimorphite-dl` | Alternative pH protonation |
| `ovito` | `ovito` | Embedded OVITO viewport |
| `md` | `lammps` | LAMMPS Python module: `md smd --run` and quick checks of SMD setups |
| `excel` | `openpyxl` | `.xlsx`/`.xlsm` input (CompTox dashboard exports) |
| `editor` | `PySide6` (its Qt WebEngine) | The Sketch dock: Schrödinger's 2D Sketcher, downloaded on first use (below) |
| `all` | ml, xtb, protonate, md, excel, interop | (not `editor`: Qt WebEngine is a large download) |

Notes on the new extras:

- **`md`.** PyPI has `lammps` wheels for Windows (win_amd64), Linux and macOS. The wheel is a CPU
  build with OpenMP (the Linux wheel tested here has no GPU package). For production pulls, and for
  the GPU package (OpenCL on Radeon), use the LAMMPS Windows installer's `lmp.exe`; every SMD folder's
  `README.txt` lists the exact commands, e.g. `lmp.exe -sf gpu -pk gpu 1 neigh no -in in.smd`.
  QuantaBench only writes the input files and parses the outputs, so the extra is optional.
- **`excel`.** openpyxl is pure Python. Legacy `.xls` files additionally need `xlrd`; otherwise save
  them as `.xlsx` or `.csv`.
- **`editor`.** The Sketch dock (Build > Sketch a molecule...) runs
  [Schrödinger's Sketcher](https://github.com/schrodinger/sketcher) (BSD-3-Clause) in Qt WebEngine.
  QuantaBench needs only `PySide6-Essentials`; Qt WebEngine is in `PySide6-Addons` up to PySide6 6.11
  and in `PySide6-WebEngine` from 6.12, and the `PySide6` package brings whichever its version has:

  ```bash
  pip install -e ".[editor]"            # in the checkout's .venv
  quantabench-cli sketcher --install    # optional: fetch the editor now instead of on first use
  ```

  The editor itself (17.5 MB zip, release v2026-4) is downloaded the first time the dock opens, checked
  against its pinned sha256 and unpacked into the programs folder (`~/Library/Application
  Support/QuantaBench/programs` on macOS, `%LOCALAPPDATA%\QuantaBench\programs` on Windows,
  `~/.local/share/QuantaBench/programs` on Linux), with Schrödinger's licence next to it. Offline, download
  the zip by hand (the dock and `quantabench-cli sketcher` print the URL) and run
  `quantabench-cli sketcher --install --from-zip FILE`, or put the zip into the programs folder.
  `quantabench-cli sketcher --use DIR` (or `QUANTABENCH_SKETCHER_DIR`) points this machine at a self-built
  version instead. `QUANTABENCH_NO_SKETCHER=1` keeps Qt WebEngine from loading at all.

## Settings in a synced folder

```powershell
quantabench-cli settings --set-dir "~/SyncThing/QuantaBench/Settings"
```

This writes a git-ignored `settings_location.json` pointer in the repo, so every computer that clones
the repo and has the SyncThing folder shares the same settings. The GUI offers the same choice under
Settings > Folders, and the install scripts under `--settings-dir` (`-SettingsDir`).
