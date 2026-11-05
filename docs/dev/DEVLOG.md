# Development log

Newest entries at the bottom. One entry per working session or nightly run.

## 2026-10-22: 0.1.0 scaffold
- Geometry builder pipeline, PySide6 GUI (OVITO-style docks), CLI (`doctor`, `build`, `slab`,
  `settings`), resource manager, graphite surface builder. 35 tests.

## 2026-10-23: 0.2.0.dev0 (multi-agent build)
- Six parallel tracks: 3D viewer, 2D viewer, resumable batches + CompTox/Excel input, frequencies +
  thermochemistry + IR, batched conformer refinement, LAMMPS rigid-adsorbate steered-MD bridge.
- Integrated into the GUI (2D|3D split with shared selection, Rotate/Select/Move modes, surface builder
  move mode + LAMMPS export) and CLI (`build --run-dir/--resume/--chunk/--freq/--batched`, `status`,
  `merge`, `freq`, `md smd`, `md parse`).
- Three independent reviews (science, GUI/threading, robustness): 24 findings (0 critical, 4 high),
  all confirmed and fixed. 176 tests, ruff clean (Linux, CPU only).
- Not verified: anything on a GPU, Windows, or with real AIMNet2 weights (see ROADMAP 0.2 "Still open").
- Planning: `docs/UI_PLAN.md` (task-oriented workspaces, job model) and `docs/dev/BACKLOG.md`.
- Next: UI-1 app shell.

## 2026-10-27: first native Windows run on BTM NOVA
- Pushed 0.2.0.dev0 to GitHub from BTM NOVA (fast-forward from a git bundle; the cloud session has no
  push access to the repo yet).
- Native Windows test run (Python 3.12.10, CPU PyTorch 2.14.1, Radeon RX 9060 XT OpenGL 4.6):
  - pyvistaqt 0.13 crashed (access violation in `initializeGL`) on the first 3D viewer → pinned
    `pyvistaqt<0.13`; with 0.12 all 31 GUI/3D tests pass.
  - dxtb dipole integrals need libcint, which has no compatible Windows build → polarizability and IR
    intensities now degrade gracefully (GFN1 frequencies/thermo still work); new backlog item WIN-1.
  - A dark-mode test assumed a light desktop → fixed.
  - LAMMPS counter-ion test lost atoms on Windows (3/6 seeds; Linux 0/12): Na+ collapsed onto the
    carboxylate with UFF LJ → Joung-Cheatham ion parameters, nve/limit and 20 fs Langevin for ions.
    Remaining: the Windows LAMMPS wheel fails ~5-10% of *identical* counter-ion runs (not threading:
    1/20 with OMP_NUM_THREADS=1); those two tests are xfail on Windows. Backlog MD-1.
  - Full suite natively on Windows: 177 passed.
- Smart App Control no longer blocks RDKit. ROCm PyTorch still to install (HW-1).

## 2026-10-28: ROCm PyTorch on BTM NOVA
- Installed AMD's ROCm 7.2.1 SDK + PyTorch 2.9.1 (Windows): `torch.cuda.is_available()` True, RX 9060 XT,
  6.1 TFLOP/s fp32 matmul. The ROCm SDK calls `offload-arch.exe` with an unquoted path, which the space in
  "GitHub Projects" splits (harmless warning so far).
- Engines on the GPU: GFN1-xTB fails (no MAGMA: `torch.linalg.cholesky` on GPU), AIMNet2 fails (NVIDIA Warp
  D3 rejects the ROCm device). CPU: GFN1 0.30 s and AIMNet2 0.28 s per energy+forces for the PFOA anion.
- AMD's wheel has a slow CPU BLAS (52 GFLOP/s vs 881 with PyPI's CPU torch); `doctor --bench` ran > 50 min.
- New backlog items GPU-1 (AIMNet2 D3 via tad-dftd3), GPU-2 (dxtb linear algebra), PERF-1 (two venvs).

## 2026-11-01: original plan's engines (multi-agent build)
- Compared the first QuantaBench plan (PyQt6, CREST, GFN2/GFN-FF, MOPAC, Psi4 + cubes, PyQtGraph spectra,
  cclib, one conda env) with the roadmap; built the missing pieces on a new shared contract
  (`chem/external.py`, engine plugins, `chem/solvents.py`, CREST/QM/panel hooks), then five parallel
  builders: xtb program + CREST, tblite + MOPAC, Psi4 QM stage, cubes/surfaces/ESP panel, spectra panel +
  cclib/JCAMP import. 366 tests pass on Linux with the real programs.
- Environment findings: Psi4 1.9.1 segfaults with libint 2.13 (pin 2.9); pint 0.26 breaks qcelemental
  0.30 (pin <0.25); conda's xtb 6.7.1 and tblite-python 0.8 cannot share an env (solver drops xtb to
  6.4.1); tblite's PyPI wheel works. (Correction 2026-11-04: xtb 6.7.1, CREST 3.0.2, MOPAC 23.2.5 and
  Psi4 1.9.1 with the pins all solve for osx-arm64; the first dry run lacked CONDA_OVERRIDE_OSX.)
- Science notes: xtb reports ion dipoles about its own origin (fixed: centre of mass, and a fresh single
  point after `--opt`); xtb's "Mol. α(0)" is the D4 model value (now `gfn2_d4_polarizability_A3`), not
  the electronic response that tblite/dxtb/Psi4 give; MOPAC energies are ΔHf.
- Follow-ups: ENG-1 (Windows), ENG-2 (CREST via WSL), ENG-3 (refactors), VAL-2 (Spartan comparison).
  Suggested `vibrations.from_hessian(hessian_ev_A2, numbers, positions, energy_ev, *, method, charge,
  multiplicity, dipole_derivative, ir_source, hessian_method, temperature, ...)` for ENG-3.

## 2026-11-04: cross-platform and two-machine plan
- New `docs/PLATFORMS.md`: platform matrix for Windows 11 / WSL2-Linux / macOS (Apple Silicon), roles for
  BTM NOVA (GPU, always on) and the MacBook (CPU QM, native CREST), a phased worker design over Tailscale +
  SyncThing (M0 to M4), and the fork/upstream plan (FORK-1 to FORK-9). ROADMAP gained a platform track
  (P1 to P3); BACKLOG gained PLAT-1/2, DIST-1 to DIST-4, FORK-1, FORK-2, FORK-5, FORK-7 to FORK-9
  (FORK-3 = GPU-1, FORK-4 = GPU-2, FORK-6 needs no work). CI runs on macos-14 (non-blocking for now).
- Facts checked: MPS has no float64; Warp is CUDA-only, CPU-only on macOS; tad-libcint 0.4.0 (2026-10-20)
  ships Windows and macOS-arm64 wheels but dxtb 0.4 pins tad-mctc 0.7; PySCF has no native Windows
  support; Ray multi-node on Windows is experimental.

## 2026-11-05: work splitting (lanes, machine registry, workers) and the platform forks
- Eight parallel builders (all on branches from the `quantabench.dist` contract commit), merged with
  integration fixes: `JobStore.record_crash`, science fields for solvent/CREST/QM, thread caps that no
  longer import torch, one `--weights` rule set for `build --chunk-weights` and `grid build`
  (`plan.parse_weights`), shared CLI argument groups, float64 AIMNet2 energies in batched refinement,
  `resources.linalg`/`dxtb_gpu` settings (also carried to lanes and workers), Machines dock.
- Measured on BTM NOVA: lanes MMFF 43.4 -> 11.7 s (24 PFAS), GFN1-xTB 172 -> 49.9 s (12 PFAS, 6 x 2);
  AIMNet2 on the RX 9060 XT through the Warp-free path, 8 PFOA conformers 5.7 s vs 59.8 s on the CPU;
  dxtb on the GPU with routed Cholesky matches the CPU but is 5-8x slower (host syncs in dxtb's overlap
  integrals), so it stays on the CPU by default.
- Found a stock aimnet 0.2 bug: `AdaptiveNeighborList` shrinks after small molecules and waits for an
  overflow error nvalchemiops 0.5 never raises, so larger molecules later in the same process lost
  neighbors (PFOA 3.2 eV, PFOS 7.6 eV off on NOVA's CPU). Fixed by `aimnet_torch.guard_neighbor_lists`.
  Earlier multi-molecule AIMNet2 CPU runs should be redone. Upstream issue/PR drafted, not posted.
- Forks: `brandonbtm29/dxtb` (tad-mctc 0.11 + tad-libcint 0.4; CI green on Linux, Windows, macOS) and
  `brandonbtm29/tblite` (Windows wheels, release `qb-v0.8.0-win1`). Upstream PR texts drafted, not posted.
- Left for Brandon: first MacBook install and a NOVA + MacBook `grid build` over Tailscale; disable the
  upstream workflows that run in the tblite fork; approve the upstream PRs.
- Integration on NOVA (2026-11-05, after two review-fix rounds): 587 passed, 41 skipped, 2 xfailed, 2 xpassed
  in the `.venv` (ROCm torch, PyPI dxtb, no tad-libcint or tblite). Found there: AMD's Windows ROCm runtime
  hangs at process exit after GPU use (pytest printed its summary, then spun a core); fixed with a hard exit
  at QuantaBench's entry points (FORK-11). Also two host-dependent tests and a third Windows-flaky LAMMPS run.
  End-to-end checks: `doctor` shows AIMNet2 on the Warp-free path and dxtb on the CPU for cuda:0; 8 PFAS with
  MMFF 14.6 s (1 lane) vs 9.0 s (`--lanes auto`, 12 lanes, mostly start-up); PFBA + PFOA with AIMNet2 on the
  GPU in 15.5 s including model load.

