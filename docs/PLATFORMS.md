# Platforms and multi-machine plan

Goal: **one QuantaBench on Windows 11, WSL2, Linux and macOS (Apple Silicon)**, where every machine can
be both a workstation (the GUI) and a worker. That lets Brandon's MacBook CPU and BTM NOVA's GPU and
12-core CPU work on the same project. When a package does not run on an OS, the plan says which route
fixes it: an upstream pull request, a fork or our own build, or sending that job to a machine where it
runs.

This file is the plan. The phases are in [ROADMAP.md](ROADMAP.md) ("Platform and multi-machine track")
and the work items are in [dev/BACKLOG.md](dev/BACKLOG.md) (`PLAT-*`, `DIST-*`, `FORK-*`). Status as of
2026-11-05. How to run workers: [WORKERS.md](WORKERS.md).

## 1. Where each piece runs today

✅ tested · 🟡 available, not yet tested here · 🔴 blocked (reason) · ↗ route the job to another machine

| Component | Windows 11 native | WSL2 / Linux | macOS (Apple Silicon) |
|---|---|---|---|
| GUI: PySide6, PyVista + pyvistaqt 0.12, pyqtgraph | ✅ (pyvistaqt 0.13 crashes on AMD: pinned) | ✅ | ✅ test suite (191 GUI tests); by hand 🟡 |
| RDKit, ASE, pipeline, batches | ✅ | ✅ | ✅ |
| PyTorch CPU | ✅ (PyPI CPU build is fast) | ✅ | ✅ |
| PyTorch GPU | ✅ ROCm 7.2.1 on the RX 9060 XT; no Cholesky/lstsq on the GPU (no MAGMA): `torch_compat` routes them | 🟡 ROCm on WSL2 fragile for this card | ✅ MPS: float32 only, no float64 ([1](#sources)) |
| dxtb GFN1 | ✅ CPU; ✅ GPU with routing (FORK-4), but 5-8x slower than the CPU, so off by default | ✅ CPU | ✅ CPU; MPS routed like ROCm (untested) |
| dxtb GFN2, multipoles, IR (libcint) | ✅ with the dxtb fork + tad-libcint 0.4 (FORK-1) | ✅ (`tad-libcint<0.4`, or the fork) | ✅ with the dxtb fork + tad-libcint 0.4 |
| AIMNet2 | ✅ CPU; ✅ GPU through the Warp-free `aimnet_torch` (FORK-3), ~10x the CPU | ✅ CPU · 🟡 CUDA | ✅ CPU; ✅ `mps` through `aimnet_torch` (Warp is CPU-only on macOS, [2](#sources)): within 1.6e-5 eV of the CPU, slower than CPU lanes |
| tblite (in-process GFN2) | ✅ wheels from the tblite fork's release (FORK-2) | ✅ | ✅ PyPI wheel (macOS 15+); same numbers as Linux and Windows to 1e-12 eV |
| xtb program 6.7.1 | 🟡 conda-forge / release zip | ✅ | ✅ conda-forge osx-arm64 ([3](#sources)) |
| CREST 3.0.2 | 🔴 no build: WSL (`wsl:`) or ↗ | ✅ | ✅ conda-forge osx-arm64 ([4](#sources)) |
| MOPAC 23.2.5 | 🟡 conda-forge / installer | ✅ | ✅ conda-forge osx-arm64; PM7 polarizability ≠ Linux's (WIN-2) |
| Psi4 1.9.1 (+ PCMSolver, dftd3/4) | 🟡 conda-forge win-64 | ✅ | ✅ conda-forge osx-arm64 |
| PySCF (future: NMR, GPU DFT) | 🔴 no native support ([5](#sources)): WSL or ↗ | 🟡 | 🟡 |
| LAMMPS Python wheel (CPU) | ✅ but intermittently non-deterministic (MD-1) | ✅ | 🟡 needs PyPI's `mpich` (`lammps[mpi]`, in the `md` extra since 2026-11-06) |
| LAMMPS GPU | 🟡 OpenCL `lmp.exe` (~9x over 12 cores in earlier tests) | 🟡 | 🟡 OpenCL build, mixed precision → FORK-8 |
| OpenMM (0.4 MD) | 🟡 OpenCL | 🟡 CUDA/OpenCL | 🟡 OpenCL; Metal plugin is beta ([6](#sources)) |

The conda environments in `envs/` solve for linux-64, win-64 and osx-arm64; CREST is the only program
with no Windows build. (An earlier note said xtb 6.7.1 and CREST were missing for osx-arm64. That was
wrong: the cross-platform dry run had no macOS version set. With one, all of `envs/tools.yml` and
`envs/psi4.yml` solve.)

## 2. Machine roles

| | BTM NOVA (Windows 11, Ryzen 9 7900X 12C/24T, RX 9060 XT 16 GB) | MacBook (Apple Silicon) |
|---|---|---|
| Best at | GPU work: batched ML potentials and dxtb (after FORK-3/FORK-4), LAMMPS GPU (OpenCL); long unattended batches (always on) | CPU-heavy QM: CREST natively, Psi4/PySCF DFT; portable GUI |
| Can't do natively | CREST, PySCF (WSL works) | float64 on the GPU; CUDA/ROCm-only codes |
| As a worker | always on | only while awake and on power (`caffeinate` during jobs) |

The scheduler should not hard-code these roles. Each machine reports what it can run and how fast
(DIST-1), and the work is split from that report.

## 3. Using both machines together

All traffic goes over **Tailscale** (Brandon already uses it; both machines need to be on the tailnet). Results go through the
existing **SyncThing** `QuantaBench/Runs` folder, under the rule already in ARCHITECTURE.md: one
machine writes a given run folder, the others only read it.

| Phase | What | How |
|---|---|---|
| **M0: works today** | Split a batch by hand | `build --chunk 1/2` on NOVA and `--chunk 2/2` on the Mac, each writing its own run folder in SyncThing, then `merge` |
| **M1: capability report** (DIST-1, done) | Each machine publishes `machines/<name>.json` in the synced settings folder: OS, CPU/GPU, engines and programs available, `doctor --bench` throughput per engine | `build --chunk-weights auto` splits a list in proportion to throughput instead of 50/50 |
| **M2: workers** (DIST-2, done) | `quantabench-cli worker` on each machine. The GUI or CLI submits jobs to it: inputs + `BuildConfig` + stage | Small HTTP/JSON service bound to the Tailscale address only, with a shared token from settings. The worker runs `jobs.run_resumable` into its own run folder and reports progress. It refuses jobs from a different QuantaBench version, and records its environment in `summary.json` |
| **M3: GUI and routing** (DIST-3, Machines dock done) | "Machines" dock and "Run on: this machine / NOVA / Mac / auto" in the job dialog (UI-5) | An auto scheduler picks the machine per job from M1's report |
| **M4: stage routing** (DIST-4) | Different stages of one molecule on different machines, e.g. CREST + Psi4 on the Mac, GPU refinement + LAMMPS on NOVA | The handoff is the run folder's artifacts (conformer SDF, cubes); the job table records where each stage ran |

**On one machine** (`dist.lanes`, with any of the phases): `build --lanes auto` runs molecules in parallel
worker processes that share the CPU threads and the GPU (for example `gpu:1x4,cpu:5x2`: one GPU lane with 4
threads and five 2-thread CPU lanes). The parent process stays the only writer of `jobs.sqlite`. On NOVA,
MMFF ran 3.7x and GFN1-xTB 3.45x faster than one process with all 12 threads.

`auto` plans a GPU lane for a GPU-capable engine only where this machine's benchmark says the GPU is the faster
lane: `<method>@<gpu>` beats `<method>@cpu` in the registry (`machines --refresh --bench aimnet2`). Both are
timed one molecule at a time in one process with the same threads, so this compares one lane with one lane; a
GPU lane slower than a CPU lane only takes four threads from the CPU lanes. The plan's reason gives the
numbers:
- MacBook (AIMNet2 8,249/h on the 14-core CPU, 5,305/h on MPS in the registry on 2026-11-06): CPU lanes only
  (6 x 2 threads with the default 13 of 14 threads), "no GPU lane: benchmark aimnet2@mps 5,305/h vs @cpu
  8,249/h, so a GPU lane would be slower than a CPU lane".
- NOVA (AIMNet2 about 10x faster on the RX 9060 XT): 1 GPU lane with 4 threads + CPU lanes on the other cores,
  "(benchmark aimnet2@cuda:0 ... vs @cpu ...)".
- No benchmark of the method on both devices: the GPU lane as before, "(no benchmark ... on this machine: GPU
  lane by default)".

A run in water reads only water benchmarks: `machines --bench tblite-gfn2 --solvent water` records
`tblite-gfn2+water@cpu`, never mixed with the gas-phase `tblite-gfn2@cpu` (methods without a water model are not
timed in water: their runs are gas phase). `--bench-set mixed` adds ethanol, toluene, caffeine, atrazine and
cholesterol (74 atoms) to the five PFAS anions; the set is recorded with each number, and the speed-aware
lanes convert it with that set's mean size.

Lanes of different speed are **speed-aware** at the end of a batch (`dist.speed`, DIST-5): a free lane does
not start the next molecule when the lanes clearly faster than it (1.5x or more) would finish every molecule
still queued before it finished that one. With one GPU lane 10x faster than each CPU lane, the CPU lanes leave
the last ~10 molecules to the GPU lane; with two such GPU lanes, the last ~20. Speeds are seconds per atom per
kind of lane, measured as molecules finish; before that, this machine's benchmark (`quantabench-cli machines
--refresh --bench aimnet2` measures `aimnet2@cpu` and `aimnet2@cuda:0`) gives the ratio, and without one the
first molecules go out as before. A lane that dies releases what was held for it at once; one that stalls,
after about one slow-lane molecule. The log says when lanes start holding back and, at the end, the speed of
each kind of lane. Switch it off with `build --no-speed-aware-lanes` or Settings > Compute
(`resources.speed_aware_lanes`).

Two limits per lane keep one molecule from stopping a batch (see [ENGINES.md](ENGINES.md#memory-guard-for-dxtb-and-the-per-molecule-time-limit)):
- **Memory:** each lane's dxtb steps get a share of the memory available at the start (available / lanes, at least
  1.5 GB; `--memory-budget GB`, Settings > Compute > Memory per lane). A step estimated above it takes a cheaper
  path (tblite for the polarizability, finite differences for the Hessian) or is skipped with a note. A refinement that does not fit fails the molecule with the reason, or makes `auto`
  try its next engine. A lane that is OOM-killed anyway is restarted and its molecule's crash counted, as before.
- **Time:** a lane still on one molecule after `--molecule-timeout` (default 1800 s; Settings > Compute > Time
  limit per molecule) is killed and restarted. The conformer embedding gets a quarter of it.
  The molecule is recorded failed with `time limit (1800 s) in <stage>`. It is not counted as a crash and not
  retried in the run; `--retry-failed` redoes it.

**Frameworks considered:**
- **Ray:** no. Its multi-node clusters on Windows are "experimental and untested" ([7](#sources)).
- **Dask distributed:** a possible later backend for M2/M3. Its worker resources can tag GPU machines.
  But it needs identical package versions everywhere and pickles Python objects between machines.
  A small QuantaBench worker keeps per-machine environments and settings independent, which is the
  whole point here.

**Platform chores:**
- Windows: allow the worker port only on the Tailscale interface.
- macOS: `caffeinate -i` while a worker has jobs.
- Both: a "pause worker" switch so the GUI machine stays responsive.

## 4. Forks, builds and upstream fixes

Policy:
- Try an upstream pull request first.
- Fork only to unblock, from a tagged release, with QuantaBench's CI testing the fork on all three OSes.
- Every fork is listed here with its upstream link, so it can be dropped once upstream catches up.
- Forks live under `brandonbtm29/` and install with `pip install "pkg @ git+https://github.com/...@tag"`.
  Wheels are attached to GitHub releases, because PyPI packages cannot depend on direct URLs.
- The pins the installers use are in `scripts/fork-packages.json` (per platform: `packages`, `only_binary`,
  `find_links`). Help > Check for updates (`quantabench-cli update`) reads the file too, and reinstalls a
  fork whose entry changed in a pull ([INSTALL.md](INSTALL.md#fork-packages-scriptsfork-packagesjson)).
  A new tag or release goes there, plus the matching install hint in `chem/engines.py` or
  `chem/eng_tblite.py`.

| ID | Problem | Route | Unblocks | Acceptance |
|---|---|---|---|---|
| **FORK-1** ✅ | dxtb 0.4 pins tad-mctc 0.7. tad-libcint 0.4 (Oct 20, 2026), the first with Windows and macOS-arm64 wheels ([8](#sources)), needs tad-mctc ≥ 0.11 | PR to dxtb supporting tad-mctc ≥ 0.11. Interim fork `brandonbtm29/dxtb` (branch `qb/tad-mctc-0.11`) with the API updates. Fallback: backport tad-libcint's Windows/macOS build onto 0.3 | GFN2-xTB, xTB polarizability and IR intensities in dxtb on Windows and macOS; replaces WIN-1 | dxtb test suite + QuantaBench xTB tests pass on all three OSes in CI |
| **FORK-2** ✅ | tblite has no Windows wheel on PyPI | PR to tblite adding Windows wheels (cibuildwheel, MinGW-w64 gfortran). Interim: build them in a `brandonbtm29/quantabench-wheels` repo and attach to releases | In-process GFN2 + ALPB on Windows in the pip venv | `pip install tblite --find-links <release>` works on Windows; tblite tests pass |
| **FORK-3** ✅ (ROCm) | AIMNet2's D3 dispersion and neighbor lists use NVIDIA Warp: CUDA only, CPU only on macOS | Adapter inside QuantaBench: load the model without dispersion, add D3(BJ) from `tad-dftd3` (pure PyTorch); then a PR to `aimnet` making Warp optional. Same work as GPU-1 | AIMNet2 on ROCm (NOVA) and MPS (Mac) | Energies and forces match the stock calculator to 1e-5 eV on CPU; runs on both GPUs |
| **FORK-4** ✅ | dxtb's dense linear algebra fails on these GPUs: Windows ROCm has no MAGMA (cholesky/eigh fail), and MPS has no float64 (its linear-algebra coverage is to be checked op by op) | Device-split wrapper in QuantaBench (tensors on GPU, eigh/cholesky on CPU, small for PFAS-size molecules); then a PR to dxtb for a configurable linalg device. Same work as GPU-2 | dxtb (and batched refinement) on NOVA's GPU and the Mac's GPU | GPU result matches CPU to float32 tolerance; measured speedup reported |
| **FORK-5** | CREST has no Windows build | Investigate a win-64 build in the conda-forge `crest` feedstock (MinGW-w64). Until then: WSL (`wsl:`) or route to the Mac (M2). Low priority | Native CREST on Windows | `crest --version` and a test conformer search on win-64 |
| **FORK-6** | PySCF has no native Windows support | No fork: route PySCF jobs (NMR, later GPU DFT) to the Mac or WSL (M2/M3) | NMR on any GUI machine | — |
| **FORK-7** | pyvistaqt 0.13's framebuffer widget crashes in `initializeGL` with the AMD driver | Upstream issue with a minimal repro; keep the `<0.13` pin; test 0.13 on macOS | Newer pyvistaqt | Repro filed; pin re-checked each release |
| **FORK-8** | LAMMPS: the Windows PyPI wheel is non-deterministic (MD-1), and macOS has no GPU build | Compare with `lmp.exe` from the Windows installer. `scripts/build_lammps_macos.sh` for an OpenCL GPU-package build in mixed precision (Apple GPUs have no fp64; mixed mode keeps double accumulation) | Reliable SMD on Windows; LAMMPS GPU on the Mac | Same SMD input gives identical results 20/20; Mac GPU vs CPU speedup measured |
| **FORK-9** | OpenMM on the Mac's GPU | Evaluate the `openmm-metal` plugin (beta, built from source) against OpenCL for the 0.4 MD bridge | MD on the Mac GPU | Benchmark and a go/no-go note |

Results so far (2026-11-05):
- **FORK-1:** [brandonbtm29/dxtb](https://github.com/brandonbtm29/dxtb), branch `qb/tad-mctc-0.11`, tag
  `qb-v0.4.0.post1`. CI green on Ubuntu, Windows and macOS. GFN2 energies, polarizabilities and IR intensities
  match PyPI's stack to ~1e-8 relative; NOVA matches Linux to 1e-15 to 4e-9.
  Install: `pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"`.
- **FORK-2:** [brandonbtm29/tblite](https://github.com/brandonbtm29/tblite), branch `qb/windows-wheels`,
  release `qb-v0.8.0-win1` (built in the fork, not a separate wheels repo). NOVA reproduces the Linux energies
  to 1e-14 Eh. Install: `pip install tblite --only-binary tblite --find-links
  https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1`.
- **FORK-3:** in QuantaBench (`chem/aimnet_torch.py`), no fork needed. RX 9060 XT within 7.5e-6 eV of the
  stock CPU calculator; 8 PFOA conformers refine in 5.7 s on the GPU vs 59.8 s on the CPU.
- **FORK-4:** in QuantaBench (`chem/torch_compat.py`). Correct on the GPU, but dxtb 0.4 is 5-8x slower there
  than on NOVA's CPU, so the default keeps dxtb on the CPU.
- Upstream pull requests for dxtb, tblite and aimnet: approved by Brandon on 2026-11-06, none posted yet.
  - dxtb: upstream main is 19 commits past v0.4.0 (GFN0-xTB, multipole integrals, CCA ordering) and still on
    tad-mctc 0.7; the change must be ported onto current main first.
  - tblite: a clean branch `pr/windows-wheels` (2 files) is pushed to the fork. tblite asks for a DCO sign-off,
    which only Brandon can give, and the `wheel.yml` change has not run yet.
  - aimnet: upstream fixed the neighbor-list bug in 0.2.5 (PR #127) and declined ROCm support (issue #3), so the
    draft is out of date; QuantaBench should move to `aimnet>=0.2.5` (needs torch >= 2.10).

## 5. Bringing macOS up (PLAT-1, PLAT-2)

1. **CI.** `macos-15` (Apple Silicon) is in `.github/workflows/ci.yml`, non-blocking until green.
   `.github/workflows/external-engines.yml` installs `envs/tools.yml` + `envs/psi4.yml` with micromamba on
   Ubuntu and macOS and runs `pytest -m external` (done 2026-11-05). Until 2026-11-06 the matrix said
   `macos-14`, which GitHub stopped running after 2026-11-05 19:15 UTC (every macOS job since was cancelled in
   2 s: "The configuration 'macos-14-arm64-us-default' is not supported"). Every GUI test was also skipped on
   macOS, in CI and on the MacBook, because the tests looked for `$DISPLAY`, which a Mac never sets.
2. **First run on the MacBook** (MacBook Pro M3 Max, 14 cores, 36 GB, macOS 27.2; 2026-11-06, at f7f1cd9):
   - Full suite with every engine installed: 841 passed, 18 failed, 65 skipped (GUI). 11 of the failures were
     the test harness (`caffeinate` as pytest's parent looked like a leftover lane process). The real ones:
     a numpy-bool `TypeError` in the spectra panel (NumPy 2 only), the macOS hostname path in
     `test_machine_name`, LAMMPS without MPICH, and the two MOPAC polarizability checks (WIN-2). With the GUI
     tests enabled, 189 of 191 passed (the two were the spectra bug). External programs: CREST, xtb, MOPAC,
     Psi4 all found in `~/miniforge3/envs`; 15 of 17 `-m external` tests pass.
   - Numbers against the other machines: tblite GFN2 single points at NOVA's geometries agree with NOVA and
     Linux to 1e-12 eV (dipoles, orbitals, polarizabilities, ALPB ΔG_solv likewise); NOVA's stored energies
     differ by up to 1.4e-7 eV only because the optimizer's last SCF started from the previous step's density.
     A full rerun of the protocol (5 molecules, gas and water) gives the same conformer counts and best
     energies within 2e-6 eV (gas) and 4e-4 eV (water), dipoles within 0.01 D; Windows and Linux x86 agree
     to 1e-12 eV on 4 of the 5. Soft modes are the fragile part: one floppy cation's lowest frequency in water
     was -37 (Linux), -33 (Mac) and -15 cm⁻¹ (NOVA), so its entropy differed by 3.3 %.
   - Still by hand: GUI checks (HiDPI picking, trackpad pinch zoom and two-finger rotate, dark mode).
3. **Mac specifics to handle:**
   - `mps` device selection with float32 only. AIMNet2 on `mps` ran without any CPU-fallback op
     (`PYTORCH_ENABLE_MPS_FALLBACK` unset, no warnings)
   - the BLAS used by conda's Psi4 (OpenBLAS vs Apple Accelerate, benchmark both)
   - Gatekeeper/quarantine on conda binaries, if any
   - sleep while running
4. **Packaging** (with the 0.5 installers):
   - a `.app` bundle, notarization later
   - on every OS, the standing requirements for Brandon's apps: an **Update button** (git pull or pip upgrade, then restart) and a **launcher that hides the terminal window**: `pythonw`/GUI-script `.exe` on Windows, `.app`/`.command` on macOS, `.desktop` on Linux. Done 2026-11-05: Help > Check for updates (`quantabench-cli update`) and the launchers the install scripts create. 2026-11-06: `--upgrade-packages` (pip upgrade, PyTorch and the forks kept) and fork pins that arrive through an update (BACKLOG UPD-1); workers still need a manual restart after an update

## 6. Order of work

1. **P1 (0.3):**
   - PLAT-2 CI on macOS (workflows done; make macOS required once green); PLAT-1 first Mac run
   - M0 documented
   - ✅ FORK-3 and FORK-4 (GPU engines on NOVA; the Mac's GPU is untested)
   - ✅ FORK-1 fork (GFN2 everywhere); upstream PR pending
2. **P2 (0.4):**
   - ✅ DIST-1 capability report + weighted chunks; ✅ DIST-2 workers; ✅ lanes on one machine
   - ✅ FORK-2 tblite wheels
   - ENG-1/ENG-2 Windows external-engine checks; the first NOVA + MacBook grid run
3. **P3 (0.5):**
   - DIST-3 GUI routing (with UI-5) and DIST-4 stage routing
   - installers and launchers on all three OSes
   - FORK-5, FORK-8, FORK-9

## Sources

1. PyTorch forum, "Float64 tensors on MPS" (June 2026: no native float64 on MPS): https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
2. NVIDIA Warp compatibility table (macOS Apple Silicon: CPU only; GPU = CUDA): https://nvidia.github.io/warp/latest/user_guide/compatibility.html
3. conda-forge xtb files (6.7.1 for linux-64, win-64, osx-64, osx-arm64): https://anaconda.org/conda-forge/xtb/files
4. conda-forge crest files (3.0.2 for linux-64, osx-64, osx-arm64): https://anaconda.org/conda-forge/crest/files
5. PySCF install guide ("not supported natively on Windows"): https://pyscf.org/user/install.html
6. openmm-metal (beta Metal platform for OpenMM): https://github.com/bdenhollander/openmm-metal
7. Ray installation ("multi-node Ray clusters [on Windows] experimental and untested"): https://docs.ray.io/en/latest/ray-overview/installation.html
8. tad-libcint on PyPI (0.4.0, Oct 20, 2026; Windows, Linux, macOS arm64 wheels): https://pypi.org/project/tad-libcint/
9. dxtb on PyPI (0.4.0; libcint interface Linux-only): https://pypi.org/project/dxtb
10. Dask distributed worker resources: https://distributed.dask.org/en/stable/resources.html
11. LAMMPS GPU package (OpenCL, mixed precision): https://docs.lammps.org/Speed_gpu.html
