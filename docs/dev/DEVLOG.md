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

## 2026-10-27: first native Windows run on the Windows workstation
- Pushed 0.2.0.dev0 to GitHub from the Windows workstation (fast-forward from a git bundle; the cloud session has no
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

## 2026-10-28: ROCm PyTorch on the Windows workstation
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
  the Windows workstation (GPU, always on) and the MacBook (CPU QM, native CREST), a phased worker design over Tailscale +
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
- Measured on the Windows workstation: lanes MMFF 43.4 -> 11.7 s (24 PFAS), GFN1-xTB 172 -> 49.9 s (12 PFAS, 6 x 2);
  AIMNet2 on the RX 9060 XT through the Warp-free path, 8 PFOA conformers 5.7 s vs 59.8 s on the CPU;
  dxtb on the GPU with routed Cholesky matches the CPU but is 5-8x slower (host syncs in dxtb's overlap
  integrals), so it stays on the CPU by default.
- Found a stock aimnet 0.2 bug: `AdaptiveNeighborList` shrinks after small molecules and waits for an
  overflow error nvalchemiops 0.5 never raises, so larger molecules later in the same process lost
  neighbors (PFOA 3.2 eV, PFOS 7.6 eV off on the workstation's CPU). Fixed by `aimnet_torch.guard_neighbor_lists`.
  Earlier multi-molecule AIMNet2 CPU runs should be redone. Upstream issue/PR drafted, not posted.
- Forks: `brandonbtm29/dxtb` (tad-mctc 0.11 + tad-libcint 0.4; CI green on Linux, Windows, macOS) and
  `brandonbtm29/tblite` (Windows wheels, release `qb-v0.8.0-win1`). Upstream PR texts drafted, not posted.
- Left for Brandon: first MacBook install and a workstation + MacBook `grid build` over Tailscale; disable the
  upstream workflows that run in the tblite fork; approve the upstream PRs.
- Integration on the workstation (2026-11-05, after two review-fix rounds): 587 passed, 41 skipped, 2 xfailed, 2 xpassed
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
- The workstation's `%TEMP%\QuantaBench` held 479 GB after b03's ESP pass: 59 `psi4_<id>_*` folders, each with a 13-21 GB `psi.<pid>.97`.
  - That file is DiskDFJK's three-index integrals: the output shows `Memory [MiB]: 2574`, `Algorithm: Disk`, and "-281 MiB Core".
  - The 4 GB per job was too little for 784 basis functions and 4,901 auxiliary.
  - The lane was killed at the time limit, so `work_dir` never ran its cleanup, and its except branch kept failed folders whole anyway.
- Owner records (pid + process start time + host) make "left behind" decidable without guessing from ages. A pid reused after a reboot has another start time.
- The test suite now points the scratch root at a temp folder (`QUANTABENCH_SCRATCH_DIR` in conftest). Otherwise the sweep at run start would have reached the real scratch folder of the machine running the tests.

## 2026-11-13: conformer checks pilot; `conformers compare` reads grid shares (afternoon task, branch `afternoon/2026-11-13-compare-grid-shares`)
- The pilot (10 tranche molecules) compared the tranches' conformers with a 200-embedding rebuild and with Spartan's MMFF search. Results are in `Runs\conformer-checks-2026-11-13\report.md`, not here.
- Pointed at a tranche's gas phase folder, `conformers compare` would have missed 4 of the 10: the MacBook ran them, and its shares (78 to 547 molecules of a tranche's ~1,000) are in the grid folder's `parts/bench-mac[-<job id>]/<run folder>` (downloaded) and synced into `Runs\`. The pilot worked around it by copying the molecule folders; `read_sources` now follows `grid.json`.

## 2026-11-13: DESC-3 feature builder (branch `feat/feature-builder`)
- Brandon's question was whether ratios of α, μ and V, made dimensionless, would predict σ better. In his QSAR study (results in `Runs\qsar-2026-11-13\report.md`, not here), the groups of `chem.liquid` plus the *reduced target* σ* = σ·v_m^⅔/kT were what helped: the target more than the features.
- The Lorentz-Lorenz group needs an optical-like polarizability: GFN2-xTB's α pushes (4π/3)α/v_m towards 1. Bosque's α gives n_D within 0.02 for hexane (1.380 vs 1.375), perfluorohexane (1.251 vs 1.252) and benzene (1.518 vs 1.501).
- The table's presets are formula columns, not hidden code, so the user can read and edit them. `tests/test_liquid.py` checks them against `chem.liquid` on random inputs, so the two cannot drift apart.

## 2026-11-13: UI-2 project documents (branch `feat/project-documents`, stacked on `feat/feature-builder`)
- A project is a folder, not a single file: its run folders (job tables, per-molecule results) are the bulk, and they already live in folders. `project.json` and `molecules.jsonl` are the index. `molecules.jsonl` is one line per molecule so a 14,000-molecule list is a few MB and reads in about a second.
- Opening restores only the light rows, as File > Open run folder does. Statuses come from the saved rows at once; the job tables are read after, off the GUI thread, because a command-line resume may have moved them on since the save.
- The table model is lazy (it refreshes only while the Table or Plot is in view): a test that adds a formula column must show Analyze first.
- Quit asks about unsaved changes only with a project open. Asking for every untitled list would have changed every existing test, and the habit of closing a session whose results are in run folders.

## 2026-11-13: QSAR-4 Model... in the Analyze table (branch `feat/model-builder`)
- The σ study of the afternoon ran as a script; the dialog makes its best setup (the liquid groups and σ*, k-fold and leave-class-out) a few clicks on any table. Its numbers stay in `Runs\qsar-2026-11-13\`, not here.
- Ridge needs no scikit-learn: with the SVD of the standardized features, the leave-one-out residual of every α on the grid is closed form (residual / (1 − leverage), the intercept's 1/n included). It picks the same α as scikit-learn's RidgeCV to 1e-15 on a test set. scikit-learn stays optional (CI runs without it; those tests skip).
- Predictions come back as a joined CSV rather than formula columns: a random forest has no formula, and a file in the project's `models/` keeps the out-of-fold predictions that the 1:1 plot needs, stored relative to the project like any join.
- The evening's machines: the MacBook took the large (> 35 heavy atoms) and iodine parts of b03's ESP pass, one molecule at a time; the workstation kept the small ones.

## 2026-11-14: RESUME-1, a resume keeps the run's options (branch `fix/resume-run-options`)
- b03's ESP pass is done on both machines. The workstation: small part, 795 ok and 54 failed. MacBook: large 56/56 and iodine 25/25 (def2-SVP).
  - The workstation's resumed segment ran at 2 threads and 4 GB per Psi4 job. The first segment had run at 4 threads and 6 GB.
  - Its log shows why: `build --resume` restored only the science fields. `qm_memory_gb`, `qm_keep_cubes` and `molecule_timeout_s` are not science (they change no number), so they came from the settings.
  - Treating them as science would make a changed value refuse the resume. Instead they are run options: kept unless given again, and never a mismatch.
- The 54 failures (53 time-outs at 1,800 s, 1 SCF failure) split 29 in the first segment and 24 + 1 in the resumed one.
  - The first segment ran the largest molecules first, at 6 GB, so the limit caused the failures more than the resume did.
  - They run again on the MacBook with the large part's settings (`Runs\2026-11-14_b03-esp-retry-mac`).

## 2026-11-14: UI-8 collapsible sections and pop-outs (branch `feat/ui-declutter`)
- The Pipeline dock's long part was the help text, not the fields: each choice had a two- to five-line note from the method guide under it. The notes now sit behind an ⓘ next to the field. The field's row stays one line, and a note that warns (an engine not installed) shows as ⚠, so it is not lost by being hidden.
- The stage list and the page stack were two widgets for one thing. An accordion of sections keeps the stage names, statuses and a summary of each stage's settings on screen at once, with one stage's form open.
- Docks could always float (Qt's title-bar button), but nobody finds that button. View > Pop out panel lists them, and the Plot has its own ⧉. Floating docks are part of `saveState`, so each workspace keeps its own pop-outs.

## 2026-11-15: FIX-2 flaky CI tests (branch `fix/flaky-ci-tests`)
- The sampler's sensor check was `if self.sensors and not self._halt.is_set()`. A stop between the wait and that line kept a sample with the load columns but no sensor columns. On a slow macOS runner the 0.35 s test saw it. Dropping the cut sample keeps every sample's columns consistent.
- The Linux hang could not be reproduced. pytest-timeout's default signal method needs the interpreter to run a signal handler, which never happens while Qt's C++ event loop blocks (a modal dialog, a nested `exec`). The thread method dumps the stacks and exits, so the next occurrence costs 5 minutes and names the test.

## 2026-11-15: UI-5 job dialog, recipes and the Jobs dock (branch `feat/job-dialog`)
- **Every build is a job.** A queue that only the dialog used would let two builds race for `MainWindow.worker`. So `run_build`, `resume_run` and the dialog all hand a `Job` to one queue, which runs it through the window's existing `_start_worker`. The run folders, resume, cancel and the molecule list behave as before.
- **The run folder is made when a job starts, not when it is queued.** `prepare_run` returns a callable for it, so a queued job that is cancelled leaves no empty folder, and the folder's timestamp is the start.
- **The queue follows the worker through `attach`, connected before the thread starts.** A signal emitted before the connection would be lost: the queue's per-job log and counts come from the same `log`, `progress` and `molecule_done` signals as the window's.
- **Tasks are `BuildConfig` changes on top of the recipe.** Optimize and single point are one structure (`max_conformers=1`, `refine_top=1`, rank `refine`); single point adds `max_steps=0`, which is ASE's energy and forces at the start geometry. The pipeline MMFF-relaxes every embedded or seeded structure, so the single point is at that geometry: a follow-up for a geometry used unchanged.
- **Recipes use the "gfn2" alias**, the first installed of dxtb, tblite and the xtb program, so one recipe works on the workstation (tblite) and on the MacBook alike. A user recipe stores every recipe field, not the difference from the defaults, so a later change of a default does not change a saved recipe.
- **The workspace state version is now 3:** a new dock is not in an older saved state, and `restoreState` leaves it wherever it was. The saved arrangements are reset once.
- **A cancelled job is "cancelled" only if a molecule was left undone.** macOS CI runs a batch in several lanes, and the molecules in flight finish after a cancel. There, all 7 of the test's molecules finished, so the job ends "done".

## 2026-11-16: RESUME-2 a resume keeps the run's `--lanes` (branch `fix/resume-lanes`)
- **Unlike the run options, a lane spec is machine-specific.** A GPU lane on the workstation (`gpu:1x4,cpu:5x2`) means nothing on the MacBook, and a thread count fits one CPU. So the record names the machine, and only a resume there reuses it. Elsewhere the machine's own setting applies, as before.
- **Only a chosen spec is kept.** A run that used the machine's setting resumes with the setting as it is now. Someone who changed the setting in between wants the new one.
- **A choice is never overwritten by a setting.** A resume on the MacBook with its own setting leaves the workstation's `--lanes` in the record, so a later resume on the workstation gets it back.
- The column is added with `ALTER TABLE` when a table is opened for writing (as `accepted` was). Read-only opens of older tables return no record. The schema version stays 5: an older QuantaBench ignores the extra column.

## 2026-11-16: `grid build --molecule-timeout` (branch `fix/grid-molecule-timeout`)
- Found while preparing the CREST rebuild of the 1,674 flexible neutrals. `run_tranche.py` passes the CREST options to `grid build`, which stopped at `--molecule-timeout` as an unknown argument.
- The option moved from `build`'s parser into `add_build_science_args`, which `build`, `grid build` and `review` share. `config_from_args` already read it. A worker takes `molecule_timeout_s` from the job's config (it is not a resource field), so every share gets the client's value.
- **The flaky worker test on Windows (seen on #15 and #20).** The worker folds the WAL into jobs.sqlite when a job ends. SQLite deletes `-wal` only when the last connection closes, and on Windows not while another handle is open. A `GET /v1/jobs/<id>` poll that read the counts at that moment kept it. `_checkpoint` now opens and closes the store again, every 0.2 s for up to 3 s, until the file is gone. The new test holds a second connection for 0.4 s; it fails with the old single attempt.

## 2026-11-16: FIX-3 native stacks of a hung test (branch `fix/ci-hang-dump`)
- **Why the hang ran into the 30 min limit despite `--timeout=300`.** pytest-timeout's thread method is a Python thread: it needs the GIL to print the stacks and call `os._exit`. A main thread stuck in Qt's C++ code with the GIL held never gives it back. faulthandler's timer is C and still printed the Python stack, which ends at the call into Qt.
- **What the watchdog found, the first time it fired (#17's Linux run): a lock-order deadlock between Qt and the GIL.**
  - The GUI thread held the GIL inside `setVisible` → `QLayout::activate` → `QObject::connect`, waiting for a signal-slot mutex.
  - A worker's thread was ending: its `deleteLater` ran `~QObject` there, which holds that mutex while it disconnects. PySide, dropping the Python callable of a connection, waited for the GIL.
  - The fix moves the destruction to the GUI thread. `QThread.started` is connected to a `_Runner` that lives in the new thread (a closure would run in the GUI thread, where the QThread object lives: the first try did, and the new test caught it). When `run` returns, the runner moves the worker and itself home.
- **So the watchdog is another process.** It has its own interpreter and GIL, and gets "start"/"end" lines from a pytest hook over a pipe, plus end of input when pytest goes away. py-spy reads the hung process from outside: every thread's native frames, the Python frames, and which thread is "active+gil".
- **`sudo -n` on Linux and macOS.** On Linux, Yama's ptrace_scope 1 lets a process read only its descendants, and the watchdog is pytest's child. On macOS, reading another process needs root. GitHub's runners allow passwordless sudo; Windows needs neither.
- **The test hangs a process for real.** It uses catastrophic backtracking in `re` (C code that never releases the GIL), then checks the dump and the kill.
- `test_lanes.lane_processes` skips the watchdog, as it skips multiprocessing's resource tracker.
- **macOS: `sample`, not `py-spy --native`.** py-spy has no native mode on macOS ("not supported on your platform", the first CI run). There the watchdog runs py-spy for the Python frames and the GIL, then macOS's own `sample <pid> 2` for every thread's native call graph, both with `sudo -n`.

## 2026-11-16: FAIL-1 the failed molecules of the QSAR tranches (branch `fix/failed-molecules`)
- **"Bad Conformer Id" was RDKit's timeout id.**
  - `EmbedMultipleConfs` stopped by `params.timeout` before its first conformer returns `[-1]`.
  - The embedding's timing call took the whole 450 s (a quarter of the 1,800 s limit). It came back with `[-1]` and that was used as a conformer.
  - The glycosides fail only with RDKit 2026.09's small-ring torsions: without them ETKDG embeds one in about a second, and with them none in minutes, with or without random coordinates.
- **The water crashes were a tblite bug at specific geometries.**
  - They are not about threads or lanes. With 2 threads per lane the refined geometries differ in the last digits, and the crash did not happen.
  - 2447-54-3's GFN2-xTB/ALPB minimum crashes a fresh Python process in one GFN1-xTB (or GFN2-xTB) CPCM single point, on Windows and on Linux. A crash cannot be caught in the process, so the terms moved to a helper process.
  - A helper per call to `compute` would cost a Python start per molecule. One kept per process costs about 0.2 s once.
  - Lanes are daemonic `multiprocessing` processes, which may not start their own: the helper is a `subprocess`. A killed lane takes it along, since the lane kill takes the process tree.
- **SCF: damping rather than temperature.** For 1423-20-7, 1,000 cycles at tblite's default damping do not converge, and damping 0.2 does. A higher electronic temperature would change the energy; the damping only changes the path. The geometry is from an optimization step of a molecule without force-field parameters.

## 2026-11-16: CONF-AUTO CREST for flexible neutral molecules (branch `feat/crest-auto`)
- **One rule, three users.** The pipeline decides per molecule (`search_rule.decide` on the prepared structure). The
  advisor estimates per molecule from its own profile (`advisor.uses_crest`, with the same charge class and strict
  rotor count). Run-level code (lanes, dynamic threads, the cost model, telemetry) only asks whether CREST can run
  at all (`may_use_crest`).
- **The charge class is the advisor's, not the conformer checks' net charge.** The checks counted a zwitterion as
  neutral (net 0). Charged chains are what GFN-FF got wrong, so a zwitterion gets RDKit. A nitro group's N+/O- is a
  bonded pair and stays neutral.
- **Rotatable bonds: RDKit's strict count on the heavy-atom graph**, as the conformer checks (`analyze_conf.py`) and
  the advisor counted them. On the prepared molecule with explicit hydrogens every methyl would count.
- **The merged pool never compares energies across the two searches.** CREST's are GFN-FF (or GFN2) energies and
  RDKit's are MMFF94s.
  - So the refinement takes each pool's lowest `refine_top` (`ConformerSet.refine_count` = all of them) and ranks them
    at its own level.
  - With the refinement "mmff" there is no such level (it uses the pool's energies), so the merge is skipped with a
    note.
  - Duplicates against CREST's conformers are found by an RMSD on the same atom order (`AlignMol`: fast). The
    symmetry-aware `GetBestRMS` on 20 x 20 pairs of a perfluorinated chain could take minutes. A symmetric
    duplicate is refined twice and removed by the re-pruning after the refinement.
- **The Careful preset and recipe now use "auto" + merge** instead of CREST for every molecule. CREST for ions was
  the part the gas test found worse.
- **The crest-ions finding** (CREST for flexible ions in water, from the DFT conformer test) now fires only under the
  RDKit search. Under "auto" ions stay on RDKit by design, until a water test of ions says otherwise (BACKLOG).

## 2026-11-16: UI-8b the UI-8 follow-ups (branch `feat/ui-declutter-2`)
- **The ⧉ needed a title bar of our own.** Qt draws a dock's title bar with its own float and close buttons, and has no hook for adding one. `DockTitleBar` replaces it with a title, ⧉ and ✕. The title label is transparent for mouse events and the bar leaves the rest to `QDockWidget`, so dragging, docking and double-click floating work as before. Tabified docks still show it above their content.
- **The colour legend's sentence went behind the ⓘ next to Color, but the colour bar stays visible.** The bar with its −/0/+ ticks shows which colour is which, and it is the part that matters while a surface is on screen.
- **Folding the Normal modes pane has to move the splitter.** A folded section in a horizontal splitter only hides its body, so the pane would stay as wide. `_on_modes_toggled` gives the pane the header's width and restores the old sizes on opening.
- **The Results fold resizes the dock only when the Conformers tab is in front** and the dock is docked, visible and not behind another tab. It undoes the fold only if the height is still the folded one: a hand-made height is kept. `resizeDocks` takes effect after the event, so the folded height is read back on the next turn of the event loop.
- **Signals from the table's model now go to a method of the folder (a QObject), not a lambda.** A lambda outlived the window in the tests and fired on a deleted timer.

## 2026-11-16: UI-5b run a job on a worker (branch `feat/run-on-worker`)
- **One worker job per GUI job, sent the way `grid build` sends a share.** The worker API, its validation, the run
  folder and the archive are the same, so a job from the dialog is resumable on the worker. It can also be opened
  there like any other, and `grid status` tools work on it.
- **A job on a worker does not wait in the local queue.** It uses another machine, so holding it behind a local build
  would waste that machine. "Auto" decides on the follower's thread (it needs every worker's queue, a network call).
  When it picks this computer, the job is handed back to the local queue as an ordinary local job.
- **Everything that touches the network runs on a thread:** the dialog's `discover`, the submit, the polls, the
  download, and reading the run folder's rows. The GUI thread only applies the updates (`JobQueue._relay`, a queued
  signal).
- **Results:** the synced copy is used only when its `worker.json` vouches for its tables, as in `grid merge`. A
  folder that SyncThing is still filling would give a partial list. Otherwise the archive is downloaded into
  `from_workers/<worker>/`, so it never lands where SyncThing writes the worker's folder.
- **Auto weighs speed by waiting jobs:** score = the `grid build --weights auto` weight / (1 + jobs running or queued).
  A fast machine with three jobs ahead loses to a free one a third as fast. Ties go to this computer.

## 2026-11-16: FIX-4 the run lock at the end of a run (branch `fix/run-lock-release`)
- **Found from main's Windows CI after #18:** a worker job ended "failed" right after its last molecule. The job's
  error was not in the output, so the likely causes were checked one by one. The job table's checkpoint cannot raise
  (#16). The release of `jobs.lock` can: it is the last thing `run_resumable` does, and every status poll reads that
  file. On Windows a file that is open cannot be deleted.
- **Shown before fixing:** a scratch script on the workstation released the lock 3,000 times while a thread read it; 1,592
  releases raised `PermissionError`. The CI test itself passed 40 times on the workstation: the poll has to land in the moment
  of the removal, which a slow CI runner (and its antivirus) makes likelier.
- **Retry, then empty the file:** a reader holds the lock for microseconds, so trying again for up to 2 s is plenty.
  If it still fails, rewriting it as `{}` (a write is allowed while others read) makes it name no process, so neither
  this process nor another one sees a live lock. Leaving the old content would make another process on the same
  machine refuse the folder for as long as the worker runs.
- The retry runs under `_held_guard`, so no other run in this process can take the lock between the release and the
  removal and lose its own lock file to it.

## 2026-11-16: FIX-5 garbage collection in the tests' main thread (branch `fix/ci-abort-gc`)
- **What the abort showed:** a pure virtual call in `~QObject` while the GUI thread processed a DeferredDelete. That
  means an object was destroyed from two places at once. The other place was most likely Python's cyclic collector in
  a sensors thread, which was allocating hard (loading Windows' certificate store) when the main thread aborted.
- **#22's Windows CI crashed too** (heap corruption, 0xc0000374, found while the main thread collected garbage), with
  the same two sensors threads. The common thread is FIX-3's worker/runner cycle. After `_Keeper.release`, only the
  collector frees those two wrappers. PySide owns their C++ objects, so a collection in a sensors thread deletes them
  there, while their `deleteLater` events wait in the GUI thread's queue. Delivering one to an object that the other
  thread is destroying gives exactly crash 1.
- **The app already guards against this.** `gui/app.py` turns automatic collection off and collects on a GUI timer,
  for the same reason (a closed dialog destroyed in a batch thread). The tests only collected at module ends, so
  inside a module the collector could still run in any thread. Now it is off for the session, and generation 1 is
  collected after each test. The full suite took 7 min in the cloud (1,714 passed).
- **The worker/runner cycle from FIX-3** was a new way to leave Qt objects to the collector. `_Keeper` now holds both
  wrappers, so they are freed by reference counting in the GUI thread.
- **Why the job hung after the abort:** CRT `abort` → process exit → DLL detach → VTK's
  `vtkMatplotlibMathTextUtilities` destructor → `PyGILState_Ensure`. The GIL belonged to a thread that process exit
  had already ended, so this can never finish. A scratch check on the workstation (`scratch/wd_inside.py`) showed that the old
  `taskkill /F /T` kills a stuck process that started the watchdog. So the kill most likely failed because the process
  was already exiting. psutil's kill plus a "still there" log will tell next time.

## 2026-11-17: a conformer batch stopped by the time limit (branch `fix/embed-batch-stopped`)
- **Reproduced in the cloud** with gitoxin's SMILES and the run's limits (`embed(mol, 50, timeout_s=450)`): the first
  conformer took 8.9 s, the note said 39 of 50, and 451 s later "ran out of time". Without small-ring torsions one
  conformer takes 0.5 s; with them a batch can stall on a few hard seeds.
- **Why a share of the time and not a second try at the end:** RDKit's timeout stops a whole
  `EmbedMultipleConfs` call and keeps at most one conformer. The only way to have time for a top-up is to give the
  batch less than all of it.
  - 80 % leaves enough time for the fast way, and changes nothing for a batch that finishes in time.
  - The timed conformer is copied before the batch, because the batch call clears the molecule's conformers.
  - The top-up uses `clearConfs = False`, so what is there stays.
- Molecules that never hit the limit get the same conformers: `test_a_limit_that_is_not_reached_changes_nothing`
  still passes.

## 2026-11-18: telemetry of killed attempts (branch `fix/telemetry-killed-attempts`)
- **Found in the CREST rebuild's job table:** the workstation's first share had 366 ok and 26 timeout attempts.
  - Every ok attempt had its features, CPU time and 8 stage rows.
  - Every timeout had none of the features and a single stage row. Two of them timed out in `refine` and one in
    `frequencies`, so their CREST searches (over an hour each) were not recorded at all.
- **Why:** features and stage timings come from the lane's process with its result, and a killed lane sends
  nothing.
  - The parent already knows the input: the features are computed at dispatch, with the same function as the lane.
  - The parent sees every stage begin (the lane's stage messages): `RunTelemetry.stage_seen` keeps that history
    until the attempt ends, and a killed attempt's stages are built from it.
  - An attempt that ends normally still uses the lane's own stage rows (CPU, memory, conformers in and out).
- **For the cost model:** CREST stages that finished in an attempt killed later now count. The killed stage
  (outcome `timeout`, wall = the limit) is censored data, kept out of the fit as before; a censored fit could use
  it later.

## 2026-11-20: CREST through WSL in a folder inside WSL (CREST-SCRATCH-1, branch `fix/crest-wsl-scratch`)
- **The problem:** the workstation runs CREST as `wsl:.../bin/crest`, in its Windows job folder through `/mnt/c`. Every
  file CREST writes crossed WSL's 9P bridge to NTFS: about 20 MB/s of disk writes right after a CREST batch started.
- **The design** (`chem.wsl_scratch`):
  - Each job gets `/dev/shm/quantabench/<job>` when /dev/shm and WSL's MemAvailable both have 2 GB
    (`engines.crest_wsl_shm_min_gb`) plus twice the job's expected size free (64 MiB + 4 MiB per atom, generous: a
    CREST folder of a 100-atom molecule holds tens of MB at a time); else `~/.cache/quantabench/scratch/<job>`.
    Both checks matter: /dev/shm's tmpfs is half of WSL's memory, and the lanes' CREST processes use the same memory.
  - Every step is one shell script through `wsl.exe -e sh -c` (probe, prepare, fetch, remove), written as one line
    like the existing WSL wrapper. Probe and prepare are two calls before each search; WSL is running anyway.
  - `run_program(program_cwd=...)` puts the Linux folder after `--cd`. CREST's stdout still goes to the Windows log
    through the pipe, which is not 9P. The cancel and a dead lane's clean-up (`kill_recorded_programs`) kill the
    process through the `.qb_pid` in the Linux folder.
  - Clean-up: a `finally` removes the folder (success, failure, cancel, time limit). A killed process leaves a
    Windows job folder that names its twin (`.qb_wsl_dir`), so `sweep_scratch` removes the twin too. Without
    starting WSL for runs that never used it, every new CREST job removes the Linux folders whose owner (the
    Windows process, the same `.qb_owner.json`) has ended.
  - A note per molecule ("CREST scratch: WSL /dev/shm (...)"), not a telemetry column: no schema change.
- **Tests:** `tests/test_wsl_scratch.py` (23) runs the real scripts through a stand-in `wsl.exe` and `df` on
  Linux (dash, as on Ubuntu). Not verified: a real WSL. The disk writes with `auto` on the workstation are the
  check (BACKLOG CREST-SCRATCH-1).

## 2026-11-22: paste, drag-and-drop and Copy as (UI-4, branch `feat/ui-4-clipboard-dnd`)
- **Starting point:** the backlog said "clipboard, drag-and-drop and Copy as: another branch". That branch
  (`feat/clipboard`) was merged on 2026-11-07 (`fa9c962`): ChemDraw CDX/CDXML/MOL/MDLCT paste, Copy as SMILES /
  MOL / CDXML, drops of files and text. This branch builds on it; nothing was started over.
- **Paste takes File > Open's path.** Pasted molecules used to go straight into the list. Now `read_candidates`
  (worker thread) gives `read_raw`-style inputs (`meta` with the reader, 2D/3D and data fields as text), then
  `import_options.choose` and `MainWindow._import_inputs` (worker thread), as for a file. Consequences:
  - A ChemDraw drawing is 2D, so it is converted to 3D on paste (default on), and the Import Options dialog shows
    on such a paste until "Don't ask again". Same as opening a `.cdx` file; a question for Brandon in the PR.
  - A table's other columns can be posted to the molecule list (screenshot `paste_table.png`).
- **New text kinds** (`chem.interchange.detect_format` order: CDXML, MRV, SD, MOL, XYZ, table, lines):
  - XYZ: standard frames (several in a row) or bare `element x y z` lines (an ORCA `* xyz` block), atomic numbers
    allowed, extra columns ignored. Bonds by `rdDetermineBonds` for charge 0 (a radical or ion fails with a
    message); stereo from the 3D coordinates. The id is the comment's first word unless it is a number.
  - Tables: a first line with a SMILES column name (`inputs.SMILES_COLUMNS` or the Pipeline dock's SMILES column)
    split by tab, comma or semicolon; one column with a `SMILES` header works too. `_rows_to_inputs` does the
    rest, so the ID column, the SMILES preference (QSAR-ready, MS-ready) and skipped-row reports match CSV files.
- **Copy as** has seven kinds. "SMILES" is the list's own text (as entered), so it needs no structure: 14,000 rows
  copy in 0.25 s including the selection. The others read structures (`gather_inputs`: inputs in memory, else one
  job-table query per run folder) and run on a worker thread above 200 molecules. Measured for 14,000 PFOA rows
  on the cloud VM: canonical SMILES 1.8 s, SDF 3.8 s (29 MB), InChI 4.8 s, InChIKey 5.6 s.
  - XYZ: the conformer the views show for the current molecule (the conformer table's row), the best conformer
    of other built molecules (read from their folders on the worker), else the input's 3D coordinates, else the
    same ETKDG + MMFF preview the 3D view shows (said in the comment line).
  - Copy as gives extra MIME types where they exist (`chemical/x-daylight-smiles`, `-inchi`, `-xyz`, ...).
- **Right-click menu:** `MoleculeListView.menu_hooks`, a list of callables that fill the menu, so the 3D build
  mode (UI-7) or others can add entries without touching the same lines. Main window untouched.
- **Drops:** the list's viewport now takes drops itself (a `QTableView` takes none by default; the window got
  them through the parent chain before). A folder is a run folder (`jobs.sqlite` or `summary.csv`), a project
  (`.qbproj`), or else the readable files directly in it. Unknown file types go to `open_files`, whose worker
  gives the "Unsupported file type" message (or Open Babel reads them).
- **Tests:** `tests/test_gui_clipboard.py` (29) and `tests/test_interchange.py` (+6). A `_FakeClipboard` stands in
  for the system clipboard (`gui.clipboard._clipboard`), so the copy tests also run where a Windows session has no
  clipboard; the two tests on the real clipboard still skip there.
- **Brandon's answers (same day, second commit):**
  - A ChemDraw paste keeps asking (Import Options) until "Don't ask again". The dialog has a note next to that
    box and the paste log line ends with where to change it: **Tools > Settings > Import** (the Settings action is
    in the Tools menu; its tab is "Import"), one constant `gui.import_options.SETTINGS_PATH`.
  - **Copy molecules** (Ctrl+Shift+C): the selected rows as a tab-separated table (ID, SMILES as entered, Name, the
    posted field columns the list shows), as text and as `application/x-quantabench-molecules`. Paste tries that
    format first and reads it with ID and SMILES as the id and structure columns (a plain table would take a
    posted DTXSID column as the id: `inputs.ID_COLUMNS` puts it first). A copy that went through Excel is text
    only and reads as any table.
  - The key: the Analyze table already uses Ctrl+Shift+C (copy with column titles, in `SheetView.keyPressEvent`).
    A window-wide action would take the key from it, so Copy molecules is a `WidgetWithChildrenShortcut` on the
    Molecules dock. Checked under xvfb with QTest key clicks: Ctrl+Shift+C in the list copies molecules, in the
    Analyze table it still copies cells with titles, Ctrl+C in the list is still Qt's cell copy.
  - For UI-9's keymap: every action is a QAction with an `objectName` (`edit.paste`, `edit.copy_molecules`,
    `edit.copy_as.<kind>`), in `win.actions_by_name`, default keys in `clipboard.DEFAULT_KEYS` only. The Molecules
    box's Ctrl+V check now compares the key with the paste action's current shortcuts, so a rebinding moves it too.

## 2026-11-22: FIX-6, the flaky macOS test of run priority (branch `fix/fix-6-macos-priority-test`)
- **CI's failure** (#25's macOS run, 2026-11-17): `apply_to_pids([child], "low")` changed 1 process and the next
  read gave 10; the following `apply_to_pids(..., "normal")` found the child at -10 ("already there"). Nothing in the
  test can make a nice value smaller, so -10 had to be the value before the change: inherited from pytest, which
  therefore runs at -10 on the runner (inferred; the test's message now says this process's nice).
- **The race, shown on the MacBook (macOS 27, M3 Max):** a scratch script forks, execs a target and calls
  setpriority(10) once at a delay swept over 0-6 ms; 80 ms later it reads the child's nice. 297 of 1,200 changes were
  lost: those that landed 0.25-3 ms after the fork (the exec), and with a re-exec launcher also those at 4-6 ms (its
  second exec). The read right after the change gave 10 in every case but one. So macOS drops a nice value set while
  the process is inside exec(), and it reports the new value until the exec ends. Probably the exec builds the new
  image's process from a copy taken when the exec starts (not checked in XNU's source).
- **Why only CI showed it:** the failing child was named "Python". python.org's framework `bin/python3.12` (what
  setup-python installs) starts and then re-execs `Python.app/Contents/MacOS/Python` with posix_spawn(SETEXEC).
  `Popen` returns after the first exec, so the test's change could land in the second. Homebrew's python3.12 and
  python.org's 3.13 on the MacBook do not re-exec, and a change right after `Popen` held in 60 of 60 tries with each.
- **The CI failure reproduced:** a C copy of that launcher (posix_spawn with SETEXEC into Homebrew's Python.app) set
  as `sys.executable` by a scratch plugin, its re-exec delayed 6-9 ms (a slower machine). pytest ran at `nice -n 5`,
  so that a lost change shows. 6 of 1,000 runs failed with CI's message, 5 in place of -10.
- **Fix:** the test waits for its child's "up" line before changing its priority. `apply_to_pids` looks again after
  0.25 s (POSIX only: Windows has no exec) and lowers again the processes an exec put back. A live priority change can
  reach a lane or a program that is just starting, and the `sh -c` wrappers of `external.run_program` exec two or
  three times. psutil keeps the same process across an exec (same pid and start time), so lowering again works.
- **After:** 0 of 500 runs failed in the same setup. The old test (no wait) with the new `apply_to_pids` failed 0 of
  300 times, so the second look fixes it in the product too. A new test with a fake process covers the second look and
  Windows' single set.
- Found on the way: `apply_to_pids` on Linux lowers only the main thread of each process (PRIO-2).
