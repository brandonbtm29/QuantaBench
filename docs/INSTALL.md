# Installing QuantaBench

QuantaBench needs Python 3.11 or newer. **Use Python 3.12** if you want AMD GPU support on Windows,
because AMD's Windows PyTorch wheels are built for 3.12 only.

The order matters: **install the PyTorch build that matches your GPU first**, then QuantaBench.
Otherwise pip may pull a generic PyTorch that cannot see your card.

## Windows 11, AMD Radeon (native, recommended for RX 7000/9000)

Native Windows is the fastest and most reliable route for Radeon cards. Earlier testing on this project
found WSL2 GPU pass-through fragile on consumer Radeons.

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

- **dxtb's GFN2-xTB, xTB polarizability and xTB IR intensities need the dxtb fork** on Windows. They
  use dxtb's `libcint` integral backend, and the only `tad-libcint` release with a Windows wheel (0.4)
  does not work with PyPI's dxtb 0.4; install the fork as in step 4 (details in
  [dxtb's libcint backend](#dxtbs-libcint-backend-gfn2-polarizability-ir-intensities)). Without it,
  GFN1-xTB optimizations, frequencies, thermochemistry, Mulliken charges and charge-based dipoles all
  work; polarizability and IR intensities are skipped with a note, and `quantabench-cli doctor` says
  what is missing. GFN2-xTB is also available through the xtb program (`xtb-gfn2`, with ALPB
  solvation, its own optimizer, dipoles, D4 polarizability and finite-difference IR intensities):
  install `envs/tools-windows.yml`. Tested on Linux so far; the first Windows run is backlog ENG-1.
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

Not tested on a Mac yet. The steps below are expected to work; report what doesn't. See
[PLATFORMS.md](PLATFORMS.md) for the platform matrix.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                 # PyPI's macOS build includes the Apple GPU backend (mps)
pip install -e ".[all,dev]"       # tblite's wheel needs macOS 15 or newer
pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"   # dxtb GFN2, IR
quantabench-cli doctor --bench
conda env create -f envs/tools.yml    # xtb, CREST and MOPAC have native osx-arm64 builds
conda env create -f envs/psi4.yml
```

What to expect on a Mac today:
- The Apple GPU (`mps`) has no float64. GPU engines run in float32.
- AIMNet2 and dxtb need FORK-3 and FORK-4 before they run on the Mac GPU. Until then they use the CPU.
- dxtb's GFN2, polarizability and IR intensities (libcint) need the dxtb fork with tad-libcint 0.4,
  whose macOS arm64 wheel is the first one (FORK-1). Tested in the fork's CI on `macos-14`, not on a Mac
  yet. `tblite-gfn2` and `xtb-gfn2` also give GFN2.
- CREST runs natively, unlike on Windows.

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

## External programs: xtb, CREST, MOPAC, Psi4 (optional)

These are separate programs, not pip packages. Install them with conda (Miniforge) into their own
environments; QuantaBench finds them there by itself (`quantabench-cli doctor` lists them):

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
| `xtb` | `dxtb`, `tblite` (not on Windows) | GFN1-xTB (dxtb), GFN2-xTB in process (tblite; dxtb with `tad-libcint`, see [above](#dxtbs-libcint-backend-gfn2-polarizability-ir-intensities)) |
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
Settings > Folders.
