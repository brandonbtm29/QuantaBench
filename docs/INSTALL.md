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
  work; polarizability and IR intensities are skipped with a note. Use WSL2/Linux for those until
  dxtb catches up.
- **pyvistaqt is pinned below 0.13.** 0.13 replaced the native-window VTK widget with a framebuffer
  `QOpenGLWidget` that crashes in `initializeGL` with the AMD driver on Windows (verified on an
  RX 9060 XT, driver 26.10.1). 0.12 works. If you upgraded it by hand: `pip install "pyvistaqt==0.12.0"`.

- **LAMMPS with counter-ions is intermittently unstable with the Windows wheel.** Identical SMD inputs
  with a free Na⁺ fail ~5-10% of runs ("Out of range atoms"); Linux/WSL runs are deterministic. Adsorbate
  runs without counter-ions are fine. Tracked as MD-1 in `docs/dev/BACKLOG.md`.

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

## macOS (planned)

The device layer already recognizes Apple's `mps` backend, and the GUI stack (PySide6, PyVista) runs
on macOS. Engine testing and packaging are scheduled for 0.5.

## Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `ml` | `aimnet` | AIMNet2 ML potential (downloads model weights on first use) |
| `xtb` | `dxtb` | GFN1-xTB (and GFN2 with `tad-libcint` on Linux) |
| `protonate` | `dimorphite-dl` | Alternative pH protonation |
| `ovito` | `ovito` | Embedded OVITO viewport |
| `md` | `lammps` | LAMMPS Python module: `md smd --run` and quick checks of SMD setups |
| `excel` | `openpyxl` | `.xlsx`/`.xlsm` input (CompTox dashboard exports) |
| `all` | ml, xtb, protonate, md, excel | |

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
