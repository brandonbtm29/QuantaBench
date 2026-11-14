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

## 2026-11-12: UI-1 app shell (branch `feat/app-shell`)
- Task workspaces with an activity bar, one Appearance dock, a conformer playback bar, a status line with the
  selected molecule and conformer, and a first Library of run folders. Each workspace keeps its own dock layout.
- Qt notes: `QMainWindow.saveState` records only the docks that are in the layout, and `restoreState` leaves an
  unrecorded dock as it is. A workspace's default therefore takes every managed dock out, adds its own docks
  (tabbed per area) and puts the others back hidden in their home areas, so every saved state names every dock.
  Qt retires a broken-up tab group's tab bar with `deleteLater`: a script or test that spins with
  `processEvents()` outside `exec()` never deletes it and sees stale tab strips; `sendPostedEvents(None,
  QEvent.DeferredDelete)` in the wait loop (the tests do this) shows the real layout.
- `jobs.open_run` on a run folder leaves an empty `jobs.sqlite-wal` and `-shm` (SQLite's read-only WAL open); the
  Library ignores an empty -wal for "Changed", and the README's `.stignore` lines keep both out of the sync.

## 2026-11-12: QM-ESP options (afternoon task, branch `afternoon/2026-11-12-qm-esp`)
- `qm_polarizability` and `qm_keep_cubes` (BuildConfig, CLI, the GUI's QM page) for an ESP pass over many molecules,
  and the imaginary-mode steps in `psi4_driver.frequencies` (NumPy normal modes with translations and rotations
  projected out, `projected_modes`, since the driver runs in Psi4's Python without QuantaBench).
- Tested on a model triatomic held linear by symmetry (a saddle like a planar NH2 on a ring): one step along the bend
  finds the 150° minimum; and the water HF Hessian's frequencies match Psi4's to 1.5 cm-1. No Psi4 run here.

## 2026-11-13: SCRATCH-1 scratch cleanup and Psi4 memory (branch `fix/scratch-cleanup`)
- NOVA's `%TEMP%\QuantaBench` held 479 GB after b03's ESP pass: 59 `psi4_<id>_*` folders, each with a 13-21 GB `psi.<pid>.97`.
  - That file is DiskDFJK's three-index integrals: the output shows `Memory [MiB]: 2574`, `Algorithm: Disk`, and "-281 MiB Core".
  - The 4 GB per job was too little for 784 basis functions and 4,901 auxiliary.
  - The lane was killed at the time limit, so `work_dir` never ran its cleanup, and its except branch kept failed folders whole anyway.
- Owner records (pid + process start time + host) make "left behind" decidable without guessing from ages. A pid reused after a reboot has another start time.
- The test suite now points the scratch root at a temp folder (`QUANTABENCH_SCRATCH_DIR` in conftest). Otherwise the sweep at run start would have reached the real scratch folder of the machine running the tests.

## 2026-11-13: conformer checks pilot; `conformers compare` reads grid shares (afternoon task, branch `afternoon/2026-11-13-compare-grid-shares`)
- The pilot (10 tranche molecules) compared the tranches' conformers with a 200-embedding rebuild and with Spartan's MMFF search. Results are in `Runs\conformer-checks-2026-11-13\report.md`, not here.
- Pointed at a tranche's gas phase folder, `conformers compare` would have missed 4 of the 10: the MacBook ran them, and its shares (78 to 547 molecules of a tranche's ~1,000) are in the grid folder's `parts/brandons-macbook-pro[-<job id>]/<run folder>` (downloaded) and synced into `Runs\`. The pilot worked around it by copying the molecule folders; `read_sources` now follows `grid.json`.

## 2026-11-13: DESC-3 feature builder (branch `feat/feature-builder`)
- Brandon's question was whether ratios of α, μ and V, made dimensionless, would predict σ better. In his QSAR study (results in `Runs\qsar-2026-11-13\report.md`, not here), the groups of `chem.liquid` plus the *reduced target* σ* = σ·v_m^⅔/kT were what helped: the target more than the features.
- The Lorentz-Lorenz group needs an optical-like polarizability: GFN2-xTB's α pushes (4π/3)α/v_m towards 1. Bosque's α gives n_D within 0.02 for hexane (1.380 vs 1.375), perfluorohexane (1.251 vs 1.252) and benzene (1.518 vs 1.501).
- The table's presets are formula columns, not hidden code, so the user can read and edit them. `tests/test_liquid.py` checks them against `chem.liquid` on random inputs, so the two cannot drift apart.

## 2026-11-13: UI-2 project documents (branch `feat/project-documents`, stacked on `feat/feature-builder`)
- A project is a folder, not a single file: its run folders (job tables, per-molecule results) are the bulk, and they already live in folders. `project.json` and `molecules.jsonl` are the index. `molecules.jsonl` is one line per molecule so a 14,000-molecule list is a few MB and reads in about a second.
- Opening restores only the light rows, as File > Open run folder does. Statuses come from the saved rows at once; the job tables are read after, off the GUI thread, because a command-line resume may have moved them on since the save.
- The table model is lazy (it refreshes only while the Table or Plot is in view): a test that adds a formula column must show Analyze first.
- Quit asks about unsaved changes only with a project open. Asking for every untitled list would have changed every existing test, and the habit of closing a session whose results are in run folders.

## 2026-11-14: RESUME-1, a resume keeps the run's options (branch `fix/resume-run-options`)
- b03's ESP pass is done on both machines. NOVA: small part, 795 ok and 54 failed. MacBook: large 56/56 and iodine 25/25 (def2-SVP).
  - NOVA's resumed segment ran at 2 threads and 4 GB per Psi4 job. The first segment had run at 4 threads and 6 GB.
  - Its log shows why: `build --resume` restored only the science fields. `qm_memory_gb`, `qm_keep_cubes` and `molecule_timeout_s` are not science (they change no number), so they came from the settings.
  - Treating them as science would make a changed value refuse the resume. Instead they are run options: kept unless given again, and never a mismatch.
- The 54 failures run again on the MacBook with the large part's settings (`Runs\2026-11-14_b03-esp-retry-mac`).

