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

## 2026-11-22: FIX-7, the Windows worker test that leaves jobs.sqlite-wal behind (branch `fix/fix-7-windows-worker-wal`)
- **CI:** `test_archive_manifest_sync_temp_files_and_failed_streams` failed on #34's Windows run (2026-11-20), as
  before on #15, #20 and #21. The job said "done" and its run folder still had `jobs.sqlite-wal`.
- **Reproduced on the workstation:** the test alone, 1,200 runs in a worktree: 16 failed (1 of 200, then 15 of 1,000),
  all on -wal (and -shm).
- **Found with a logging copy of the test** (scratch, not in the repo; 14 of 800 failed). It records every
  `JobStore` open and close with its thread and whether -wal exists, `_checkpoint`'s start and end, and each status
  poll's `finishing` check. When -wal is found it also lists every open connection (`gc.get_objects`), the process's
  file handles and its threads. Every failure looked the same:
  - an HTTP poll checks `finishing` (False) and opens the table read-only while the job runs, 8-13 ms before the
    checkpoint;
  - the run closes its store, the worker writes the final counts and tables, and `_checkpoint`'s writable close
    removes -wal (it is gone at the checkpoint's end);
  - the poll closes its connection 2-18 ms later, and -wal and -shm are back.
  No connection was left open afterwards and no handle was on the files, so it was not a leaked connection, a
  background thread or Windows locking a temp folder. The poll's read took 12-33 ms because it shared the GIL with
  the worker's thread, which was writing the final tables.
- **Why FIX-2's `finishing` was not enough:** it stops polls that start after it is set. A poll that checked it just
  before, and was still reading, kept its connection open across the checkpoint. A read-only connection recreates
  -wal (DEVLOG 2026-11-12) and cannot remove it. On Linux the writer's close leaves -wal while another connection is
  open and `_checkpoint` tries again; on Windows the close removed it under the open reader, so nothing was left to
  retry.
- **Fix:** `Job.table_lock`. The poll's check and read are one step under it (`JobManager._live_counts`), and the
  end of the run sets `finishing` under it, so the end waits for a read in progress. Polls only wait on each other for
  the few ms a read takes.
- **After:** 0 of 1,000 runs failed, and 0 of 800 for the logging copy. The new test passed 30 of 30 times on the
  workstation.
- **Test:** `test_the_end_of_a_run_waits_for_a_status_poll_reading_the_job_table`. It holds a poll inside `open_run`'s
  last query after the run ended and checks that the checkpoint does not begin until the poll lets go. Without the
  lock it fails on Linux too.

## 2026-11-22: CREST through WSL: one BLAS thread and a bounded log (CREST-LOG-1, branch `fix/crest-log-1-openblas`)
- **The problem** (the ops agent, ~02:50 at the resume of CREST-RUN-1 with 2 lanes x 5 threads): the workstation's
  disk writes went back to 14-24 MB/s though CREST now runs inside WSL. WSL's disks wrote 6 KB/s; one lane's
  `crest.log` in the Windows job folder reached 368 MB in 11 minutes, nearly all one line: "OpenBLAS Warning : Detect
  OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option."
- **The cause, confirmed:** conda-forge's CREST 3.0.2 in WSL links libgomp and `libopenblas-0.3.34-pthreads` (its
  `conda-meta`, `ldd`). `program_env` set `OPENBLAS_NUM_THREADS` to the molecule's threads. CREST 3 runs its
  calculations in its own OpenMP threads, and a pthreads OpenBLAS called inside an OpenMP region with more than one
  thread prints that warning on every call. 45 s of CREST at `-T 5` on 544-63-8: 27,192 warning lines (3.3 MB) with
  `OPENBLAS_NUM_THREADS=5`, none (240 lines, 11 KB) with 1. CREST's install notes ask for `OPENBLAS_NUM_THREADS=1`
  "as there may be an ugly warning in the concurrent (nested) parallel code parts otherwise".
  - xtb 6.7.1 (`--opt`, `--grad --alpb water`) and tblite 0.6.0 (`run --grad`) from the same environment print no
    warning with 5 BLAS threads: their BLAS runs outside their OpenMP loops, and xtb's docs recommend
    `OPENBLAS_NUM_THREADS=<ncores>`. They keep their thread caps.
  - The Mac's CREST (the fork's build) links an OpenBLAS built with `USE_OPENMP`: no warning, and inside CREST's
    threads it already runs one thread.
- **The fix:**
  - `crest.child_env` adds `BLAS_ENV` = `OPENBLAS_NUM_THREADS=1` (natively and through WSL; `WSLENV` forwards it).
    `OMP_MAX_ACTIVE_LEVELS=1` was already set. `GOTO_NUM_THREADS` is not needed: OpenBLAS reads
    `OPENBLAS_NUM_THREADS` first.
  - The WSL wrapper sends stdout and stderr to `$QUANTABENCH_WSL_LOG` when it is set (`exec >file 2>&1` before the
    final exec, so the PID, the priority wrapper and `.qb_pid` are unchanged). `run_program(program_log=...)` sets it
    for a program in a folder inside WSL; CREST passes `crest.out`. When the program ends or is killed at the time
    limit, `wsl_scratch.output_tail` (one awk pass inside WSL) appends to the Windows log: the first 200 and the last
    2,000 lines, 1 MiB at most (a quarter for the head), lines cut at 2,000 characters, runs of identical lines as one
    line and "[QuantaBench: the line above repeated N more times]", and one line saying how much was left out.
    Above 16 MB of output the run's log gets a WARNING. A failure brings `crest.out` back whole when it is under 8 MB
    (the existing rule for small files); keep_scratch brings everything.
  - Through WSL with `crest_wsl_scratch = off`, or natively, the output still goes through the pipe (the BLAS fix
    alone stops the flood there).
- **On the workstation** (544-63-8, tetradecanoic acid, 12 rotatable bonds; GFN-FF `--quick`, `-T 5`, nice 19; other
  agents were using the machine, so single timings scatter):
  - Two lanes at once, as in the batch, same start geometry, main's way (5 BLAS threads, output through wsl.exe to
    NTFS) next to the branch's, twice: 272 s vs 184 s and 157 s vs 123 s. Main's CREST had 9 threads and kept about
    8.5 cores busy on a 5-thread lane (1,144 CPU-s vs about 500 in round 2): OpenBLAS's own threads on top of CREST's.
    Its log grew at 0.6 MB/s (165 MB and 98 MB), through about 3,500 writes a second by wsl.exe (177 bytes each).
    The branch's lane: no write by wsl.exe while CREST ran, `crest.log` 0 bytes until the end; CREST's whole output
    74-120 KB.
  - Best GFN-FF energy of the ensemble: -4150.225 / -4150.226 kcal/mol (main) and -4150.225 / -4149.484 (branch); one
    lane at a time, 5 BLAS threads -4149.592 / -4150.082, 1 thread -4150.084 / -4150.225. The batch (2026-11-20)
    found -4150.225. Same spread either way: single-threaded BLAS does not slow CREST or change what it finds.
  - The smoke job through QuantaBench with the branch (one lane of 5 threads, the batch's protocol, background
    priority): CREST 107 s (the batch: 140 s with a second lane running), 1,130 conformers, `crest.log` 0 bytes while
    CREST ran (`crest.out` inside WSL: 23 KB, 527 lines, no warning, at the last look before CREST ended), the note
    "CREST scratch: WSL /dev/shm". The lowest GFN2 conformer: -1426.75296 eV, 0.17 kcal/mol below the batch's
    (-1426.74561).
  - The machine's total disk writes could not isolate CREST: other work wrote 2-400 MB/s while these ran. One lane at a time:
    4.9 MB/s on average with the flood through NTFS, 4.3 with the same flood inside WSL, 2-4 idle; during the smoke
    job 4.1 MB/s (317 writes/s). The per-process numbers above are the clean measure.
- **Tests:** `tests/test_crest.py` (the env natively and through WSL, xtb's unchanged, `program_log` passed),
  `tests/test_wsl_scratch.py` (CREST writes `crest.out` in its WSL folder and gets one BLAS thread; the same line
  1e6 times; 1e6 numbered lines; long lines and the byte bound; the time limit; no folder inside WSL; a CREST that
  prints the warning 1e6 times (123 MB) and fails: the kept `crest.log` has 4 lines and `crest.out` stays behind).
  The scripts ran against the real WSL too (mawk; 122 MB summarized in 0.3 s).

## 2026-11-22: programs never outlive QuantaBench; probes can't hang (ENG-3, branch `feat/eng-3-process-robustness`)
- **The problem** (the 2026-11-01 review's leftovers): a GUI or CLI that crashed or was ended in Task Manager left
  xtb, CREST, MOPAC or Psi4 running, with nobody to stop them; on Windows `subprocess.run(timeout=...)` in the version
  probes could block past its timeout when a launcher's child kept the pipe open; the conda scan took the first
  environment in alphabetical order, not the newest program; the GUI's first start ran the `--version` probes on the
  GUI thread.
- **Windows** (`quantabench.proctree`, ctypes, no new dependency): `start` creates the program suspended, puts it in a
  job object of its own with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, then resumes it (`NtResumeProcess`), so nothing it
  starts escapes the job. `run_program` and tblite's CPCM helper start their programs this way; `release` closes the
  job when the program has ended (ending what it left behind), `_kill_tree` terminates the job first (it also reaches
  a child whose parent has exited, which `taskkill /T` misses). The usage meter reads the same job's accounting instead
  of making one of its own. Lanes are adopted into one kill-on-close job of their parent (`Job.adopt`; each lane's
  programs are in jobs nested in it). No other limit is set: the run priority is untouched.
- **What WSL does** (measured on the workstation, WSL 2, kernel 6.18, `scratch/eng3/wsl_probe*.py`):
  - wsl.exe starts helpers of its own: a second wsl.exe, wslhost.exe and two conhost.exe, all its children, so all in
    the job (and the "launcher's child holds the pipe" case of the probes).
  - When wsl.exe is ended, WSL hangs up the Linux session (SIGHUP): an ordinary program ends within a second or two;
    one that ignores SIGHUP (`nohup`) or runs in another session (`setsid`) does not. So the old note that killing
    wsl.exe never reaches the Linux side no longer holds, but it is not enough on its own.
  - Closing wsl.exe's stdin pipe reaches the Linux side as EOF.
- **The guard** (`external.WSL_GUARD`): a `wsl:` program runs as `sh -c WSL_GUARD sh -c WSL_WRAPPER prog args`, with
  wsl.exe's stdin a pipe QuantaBench holds open and never writes. The program runs in the background in the guard's
  process group; a watchdog subshell (ignoring INT/TERM) kills the whole group, SIGTERM then SIGKILL 2 s later, on
  WSL's hang-up or at the lifeline's EOF (QuantaBench died but wsl.exe lived on: no job). When the program ends, the
  guard stops the watchdog, sends SIGTERM to what is left in its group (CREST's xtb after a cancel: ENG-2's note) and
  exits with the program's code. `kill -TERM -$$` can only reach the guard's own group. `.qb_pid`, the cancel through
  it and the run priority's `nice`/`ionice` (in `WSL_WRAPPER`) are unchanged. `stdin_text` programs get no guard.
- **Probes** (`proctree.run_probe`): output to temporary files (no pipe to wait for), the program started like
  `start` (job on Windows, its own session on POSIX), the whole tree killed when it exits or times out. Used by
  `_probe_version`, `python_module_version`, g-xTB's start check, `md.bench`'s `lmp -h` and the WSL cancel helper.
  A program that prints its version and then hangs now gives its version.
- **Conda scan:** every environment that has the program is a candidate; the newest version wins, read from
  `conda-meta/<package>-<version>-<build>.json` (nothing runs; `ProgramSpec.conda_package` when the package name is not
  the program's: openbabel, ambertools), else the version cache; the first in the old order among equals.
- **GUI first start:** `MainWindow.__init__` builds the window inside `external.deferred_probes()` (no probe on this
  thread; an uncached version comes back as "" and MOPAC/g-xTB treat it as available for now), then `_probe_programs`
  runs `external.probe_versions` in a worker for xtb, CREST, g-xTB and MOPAC and relabels the engine lists. Every later
  start has the versions on disk and runs nothing.
- **Not done (POSIX):** no job objects there. Cancels, time limits and dead lanes were already covered (sessions, the
  lanes' program registry); a GUI or CLI killed outright still leaves a running native program until it ends. A
  lifeline guard like WSL's would work there too, but it changes which PID the usage meter samples; not needed now.
- **Tests** (`tests/test_proctree.py`, `test_external.py`, `test_gui_engines.py`): probes bounded with a child that keeps
  stdout open (all OSes); Windows: the job, a kill that reaches an orphan, a stand-in GUI killed outright with its
  engine and grandchild ending, an adopted lane, and through the real WSL a program and child that ignore SIGHUP
  ending when the GUI is killed, with the job and with only the lifeline; Linux: the guard's lifeline through a
  stand-in wsl.exe, its exit code and leftovers; the conda scan; deferred probes and the GUI's start-up worker.
- **On the workstation:** `tests/test_proctree.py` 10 passed (both real-WSL cases included); the real xtb tests (5)
  and the real CREST GFN-FF search through WSL pass. A real CREST search through WSL (decanol, GFN-FF, 2 threads)
  cancelled after 12 s, and again in a process killed outright after 12 s: nothing left running in its folder inside
  WSL (5 s after the kill). With the job switched off (the stand-in GUI's `QBTEST_NO_JOB=1`, the behaviour before),
  the stand-in engine and its grandchild outlive the killed GUI and the test fails. The full suite (offscreen Qt and VTK): 1,811 passed; the 22 failures also fail on `main` or
  pass alone (BACKLOG TEST-WIN-1).
- **Merged after CREST-LOG-1** (#47): a `wsl:` program now runs as `sh -c WSL_GUARD sh -c WSL_WRAPPER prog args`
  where `WSL_WRAPPER` also sends the output to `$QUANTABENCH_WSL_LOG` when it is set; the guard's stdout and stderr
  stay on wsl.exe's pipe. The guard's `wait` printed "Terminated" into the run's Windows log when a program was
  stopped (cancel, time limit); CREST-LOG-1's time-limit test caught it. `wait $p 2>/dev/null` keeps that line out;
  the exit code is unchanged.

## 2026-11-22: Settings > Compute, Spartan's Jobs tab (SETTINGS-1, branch `feat/settings-1-compute`)
- **What Spartan's Jobs tab has that the Compute tab lacked:** the "12 cores detected (24 logical)" line, and a short
  front page. The tab now opens with "This computer: N cores, M logical" (`settings_dialog.cores_line`: the
  detection report's `cores_text`, so Apple Silicon shows its performance cores; psutil before the report arrives).
- **The Expert box** (a folding section, closed by default and remembered per machine like the others) holds the lane
  spec, GPU linear algebra and dxtb on the GPU. Its header counts the ones not at their defaults ("2 changed", the
  tooltip names them), so a hidden non-default is still visible.
- **Lanes without the spec syntax:** *Molecules at once* lists Automatic, one at a time and 2 to the physical cores;
  any other spec (GPU lanes, threads per lane, more lanes than cores) shows as *Custom*, and choosing Custom opens the
  Expert box. Both controls edit the one value, this machine's `dist.machine_lanes` entry.
- **Same values as the job dialog and the Machines dock** (checked): the job dialog starts from
  `BuildConfig.from_settings` (`priority_for`, `free_threads_for`), the Machines dock and `machines set` from
  `priority_for(machine)` / `lanes_for(machine)` and save through `update_machine`, which stamps each change in
  `dist.machine_updated` for the SyncThing conflict merge. The dialog did not stamp, and it wrote the lane spec back
  on every OK: a lane change the Machines dock saved while the dialog was open was lost. Now the dialog writes the
  spec only when it changed, and stamps its per-machine changes the way `update_machine` does.
- The job dialog's note pointed at "Settings > Resources" (the tab's old name).
- Tests: `tests/test_gui_settings_compute.py` (5: the cores line, the Expert box and its count, the two lane
  controls as one value, a round trip read back through `BuildConfig`, `control.machine_lines` and a new dialog, and
  the Machines dock's newer values kept). Screenshots: `docs/dev/screenshots/SETTINGS-1/compute_*.png`.

## 2026-11-22: Settings > Controls, key bindings with presets (UI-9, branch `feat/ui-9-controls`)
- **Why:** Brandon asked to reconsider every key binding (without copying Spartan's for IP reasons) and to add
  Controls to Settings: a QuantaBench default, keymaps from other programs and Spartan's as options, reset and rebind.
- **The registry** (`gui/keymap.py`): every command with a key is a named `QAction` (`objectName` = its name in
  `SPECS`: title, category, default keys as `StandardKey` or portable text, scope). Scopes (window, views, 3D view,
  2D view, Molecules dock, table, Methods guide) give where a key works and which keys conflict (`scopes_overlap`).
  The modules that set keys (main window, workspaces, project, review, jobs, clipboard, pop-out, the views, the table,
  the Methods guide) bind named actions; the 3D view's VTK keys became actions with a `KeyDispatcher` on the VTK
  widget, and `dispatch_key` runs a view's keys when Qt's shortcut map did not (an inactive window, as in tests).
- **Default keymap** chosen from Qt's standard keys, Apple's HIG and Microsoft's keyboard guidelines plus
  QuantaBench's logic; written up with a full table in docs/CONTROLS.md. Changes: Open project Ctrl+Shift+O (no
  Ctrl+Alt: AltGr), Open run folder Ctrl+Shift+R, Close project Ctrl+W, Help = the platform's key, workspaces ⌥⌘1-5 on
  macOS, ⌘. cancels on macOS, plain keys only in the structure views (arrows step conformers and molecules, Space
  plays, + - zoom, Ctrl+arrows turn the 3D view, Esc / Ctrl+A on atoms), Edit > Find a molecule.
- **Presets** (key assignments only, from public docs): Avogadro 2 (avogadro.cc), VMD (User's Guide, Hot Keys),
  Spartan (the Spartan'26 manual's text: PgUp/PgDn/Home/End, Ctrl+V; its menu accelerators are only in pictures).
  PyMOL, GaussView and Maestro considered, not built.
- **Settings > Controls** (`gui/controls_page.py`, Tools > Keyboard controls...): preset with sources, searchable table,
  key capture, conflict question (Swap / Clear the other / Cancel), Clear, Reset this action, Reset to QuantaBench
  defaults, Export/Import JSON; applied on OK. Kept per computer in the window settings (modifiers differ by OS).
- **Tests:** `tests/test_gui_controls.py` (16); workspaces, viewer3d and methods-guide key tests moved to the actions.
  Windows full suite (offscreen Qt, `PYVISTA_OFF_SCREEN=true`): 1,818 passed, 20 failed: 14 `test_lanes`, 2 real-MOPAC
  and 3 GUI tests that also fail on `main` there (TEST-WIN-1), and ENG-3's Settings tab list, fixed here; the GUI files
  again after main's merges: 157 passed, the same 3. Screenshots: `docs/dev/screenshots/UI-9/`.
- **Open:** after #39 (UI-4) and #44 (UI-7) merge, bind their actions through `keymap.bind` (names reserved in
  `SPECS`); the mouse is not rebindable. Next: whichever of UI-4/UI-7 lands, then the follow-up binding.

## 2026-11-22: Settings > Build defaults, Spartan's Import Filters and Conformer Rules (SETTINGS-1, branch `feat/settings-1-build-defaults`)
- **Spartan's two settings** (from its manual): the Import Filters remove water from PDB and SD files and HCl from SD
  files; the Conformer Rules are Normal, Skeletal or Thorough, and Thorough "considers twist-boat conformers of
  six-membered rings (in addition to chair conformers)". The Import tab (UI-4) already filters files; these are the
  build's own, for every input, and science settings.
- **Build filters** (`standardize.FragmentStrip`, `standardize(strip=...)`): after the cleanup (ionic metal bonds
  disconnected, reionized), before the largest-fragment rule. Water and acids are the import filters' own
  categories (`import_options.fragment_kind`). Counter-ions are named (`NAMED_COUNTERIONS`, 34: alkali and
  alkaline-earth cations, ammonium and alkylammoniums, ethanolammoniums, halides, oxyanions, BF4-, PF6-, acetate,
  formate, trifluoroacetate, mesylate, tosylate) and match by the neutral parent, so triethylamine drawn neutral next
  to an acid is triethylammonium. Other metals are left to the metal-complex rule. The last fragment is never stripped.
- **Where they change a result:** with *Keep largest fragment* on, sodium PFOA hydrate gives the PFOA anion either
  way (only the notes differ); off, the build computes O + PFOA- + Na+ without filters, PFOA- + Na+ with water
  stripped, PFOA- with water and sodium stripped (`tests/test_build_filters.py`). With it on, a named counter-ion
  that is the larger ion is still removed: tetrabutylammonium trifluoroacetate computes the trifluoroacetate.
- **Twist-boats** (`conformers.twist_boats`): each saturated six-membered ring of the 5 lowest MMFF94s conformers is
  held at the twist-boat torsions (30, -65, 30, 30, -65, 30; three rotations and their mirror images) and optimized,
  then optimized freely. Cyclohexane: +5.9 kcal/mol (MMFF94s). ETKDG already makes some twist-boats at random
  (methylcyclohexane: three within 10 kcal/mol); on trans-decalin-1-ol the thorough pool has 17 conformers against 16
  and more non-chair rings; PFECHS keeps the same five conformers (its twist-boats fall back to chairs or lie
  above the window). Max conformers caps the thorough pool; an acyclic molecule's pool is unchanged.
- **Science fields** `strip_water`, `strip_acids`, `strip_counterions`, `ring_conformers`: defaults are the old
  behaviour, so a run stored before them resumes without a difference (`config_differences` compares a missing field
  with the default). `strip_counterions` is one sorted string ("potassium,sodium"): `BuildConfig.from_settings`
  sorts it and drops unknown names, the worker refuses unknown or unsorted ones, so two machines never differ by
  spelling. A worker from before this change refuses the fields as unknown rather than computing without them.
- The review (advisor) prepares molecules with the same filters (its cache key includes them) and lists them.
- Tests: `tests/test_build_filters.py` (12), `tests/test_gui_settings_build.py`, and
  `test_worker.py::test_build_filters_travel_to_a_worker` (a worker over HTTP). Screenshot:
  `docs/dev/screenshots/SETTINGS-1/build_defaults_tab.png`.

## 2026-11-22: Settings > Lookups, Spartan's URLs and Databases (SETTINGS-1, branch `feat/settings-1-lookups`)
- **Templates** (`quantabench.lookups`): `%inchikey`, `%inchi`, `%cas`, `%smiles`, `%name`, replaced longest first
  (`%inchikey` before `%inchi`) and percent-encoded. The structure keys are of the record as drawn, all fragments
  (potassium PFOS has its own InChIKey), which is what the databases index. The CAS RN comes from a posted field named
  like CAS, then any field, the id or the name that is a CAS RN as a whole (`pubchem.cas_numbers`, check digit).
- **Defaults**, checked where the site allowed it: NIST `cbook.cgi?ID=335-67-1` shows PFOA; CompTox
  `dsstoxdb/results?search=` is the dashboard's own search link; PubChem `#query=` is its search page. nmrshiftdb2
  documents InChI only for its data export (`exportcmlbyinchi`, 13C, `format/jcamp`), and its site refuses robots,
  so that template (in Turbine's query form) is unverified: Spartan's own NMR template would settle it.
- **Menu** (`gui.lookup_menu`): the molecule list's right-click menu has *Look up* with one action per template
  (the URL as tooltip), greyed out with "(no CAS RN)" where a value is missing, and *Edit lookups...*.
- **PubChem cache:** `ResponseCache.max_age_s` makes an answer older than the maximum age a miss (fetched again and
  replaced); offline mode reads it anyway (`stale_ok`). `default_cache_path` takes Settings > Lookups' folder after
  `QUANTABENCH_CACHE_DIR` (the tests' own) and before the platform's. A client made with its own `ResponseCache`
  keeps that cache's age unless given one.
- Tests: `tests/test_lookups.py` (6), `tests/test_gui_settings_lookups.py` (2). Screenshots:
  `docs/dev/screenshots/SETTINGS-1/lookups_tab.png`, `lookup_menu_*.png`.

## 2026-11-22: the Simulate workspace design (RES-1, branch `feat/res-1-simulate-design`)
- **Survey** of LAMMPS-GUI, VMD + QwikMD, CHARMM-GUI, OpenMM-Setup, Materials Studio (Forcite, Amorphous Cell),
  MAPS, SAMSON, Packmol/moltemplate and OVITO, plus OpenFF Interchange/NAGL and RadonPy, over system building,
  force fields, protocols, run control, live monitoring and analysis. Sources are the tools' own pages, fetched on
  2026-11-22. The commercial tools' vendor pages give feature lists only.
- **The design** (`docs/SIMULATE_DESIGN.md`) keeps the runnable folder as the product and makes MD runs jobs of
  their own kind, made of segments: one engine process each, ended by a pause, a time window, a move to the GPU or a
  crash. Pilot B's hand-written resume and the MacBook's deadline stop were the reason. LAMMPS first, OpenMM with
  OpenMM-ML in v2/v3. The lessons table uses the final reports of pilots A, B and C (2026-11-22).
- **Found on the way:**
  - LAMMPS' `fix halt` with `is_file()` and `error soft` stops a run cleanly when a file appears, so a pause can
    write a restart file without killing the process.
  - LAMMPS marks `fix smd` as unmaintained and recommends `fix colvars` or `fix plumed`: the flexible steered MD
    should be written on Colvars.
  - Binary restart files are meant for the same executable and platform only, so moving a run between machines
    needs a data file.
- **Open:** 12 questions for Brandon in the doc (DEC-1). SIM-1 stays blocked until he approves.
- **Brandon's first answers (same day), now in the doc:**
  - Splitting across machines (§6): replicas, hand-offs through data files, or GPU + CPU in one machine, with the
    trade-offs. It is v2, gated on a consistency test with pilot A; v1 stays on this computer.
  - A resource-aware scheduler (§7, SCHED-3). The pilots' LAMMPS timing breakdowns show GPU MD is not GPU-only:
    PPPM and the bonded terms on the CPU took 75-86 % of the loop time.
  - A COMSOL-like parameterization builder (§9.1): nodes per component with a resolution, global physics nodes
    with the cutoff as the "screening length", compatibility checks (combining rules, 1-4 scaling, units, a
    cutoff without a tail). Its minimal version goes into SIM-1 v1.
  - LigParGen allowed; σ by default 16 Å + tail on the GPU, with the C6F14 target Freire 2006 and no offsets.
  - Where MD runs live stays open, with the facts: run folders of 131-182 MB, and `.stignore` lines for restart
    files, the STOP file and frames.
  - New BACKLOG items BUILD-CLEAN (a "Clean up" for 2D and 3D) and SCHED-3.

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
- **Merged after SETTINGS-1's Lookups tab** (#46): `test_ctrl_shift_c_in_the_list_and_in_the_analyze_table` left Shift
  in `QGuiApplication.keyboardModifiers()` (QTest's Ctrl+Shift clicks), so `test_lookups_tab_round_trip`'s
  `selectRow` extended a selection instead of making one and removed nothing. The clipboard test now releases Shift
  and Control before it closes its window.

## 2026-11-22: the model kit's Build mode (UI-7, branch `feat/ui-7-build-mode`)
- **Built as `MODEL_KIT_GUI.md` designed it**, with the differences listed at the top of that note. The view stays a
  view: `MoleculeViewer3D` gained `set_stubs` and `buildClicked`, and `gui/model_kit.py` owns the
  `EditableMolecule`, its `QUndoStack` and the palette. The handlers are plain methods (`click_stub`, `click_atom`,
  `click_space`, `choose`, the tools); the tests call them directly, plus one test that goes through the view's
  VTK-free mouse methods to a projected stub tip, in perspective and orthographic.
- **Undo:** `UndoAdaptor.push` applies the command first and pushes only when it succeeded; the pushed
  `QUndoCommand` skips the `redo()` that `QUndoStack.push` calls at once. A `chem.edit.Command` replays its stored
  snapshot on redo, so undo/redo round trips give exactly the same coordinates (the test compares arrays with
  `np.array_equal`). The stack emits `cleanChanged` while it is destroyed with the window, after its panel's buttons
  are gone: the slot checks `shiboken6.isValid` first.
- **Stub or atom:** the design said stubs first. A stub pointing away from the viewer projects onto its own atom,
  so a click meant to replace the atom attached at the far side; the nearer hit along the ray wins now.
- **Who owns the views:** an open edit, like the Surface Builder. `MainWindow.kit_owns_view()` guards
  `_on_molecule_selected`, `_show_conformer` and `_preview_input`, so the list, the conformer table and the playback
  bar leave the model alone; `end()` gives the views back and shows the selected (or new) molecule.
- **Screenshots:** `QWidget.grab()` cannot read the VTK OpenGL widget offscreen (black). `scripts/model_kit_smoke.py`
  builds PFBA atom by atom and GenX from groups in the real window and pastes the view's own VTK screenshot into the
  window grab; `docs/dev/screenshots/UI-7/`.
- **Seen on the way, also on `main`:** a "Pipeline / Surfaces" tab bar left at the right edge of the 3D view after
  the start-up switch to Build (BACKLOG UI-1b); `test_gui_declutter.py::test_results_dock_folds_for_one_conformer`
  fails in the cloud container under xvfb on `main` as well (dock heights).

## 2026-11-22 (later): the Model Kit as QuantaBench's own design, metals, two layout findings (UI-7, PR #44)
- **Own design.** Brandon asked for the kit not to mirror another program's for IP reasons. It no longer has a fixed
  palette of hybridized atom types:
  - *element*: a common-element row, plus a compact periodic table under "Inorganic and other elements";
  - *geometry*: `edit.geometry_options` lists what the element can take at the chosen charge. VSEPR shapes come from
    the same lone-pair arithmetic as the edit model; for metals, every coordination geometry RDKit accepts that many
    bonds for. The icons are schematics drawn here: dots for bond sites, dashed rings for lone pairs, faint polyhedron
    edges.
  - *Charge* and *Bond to it*: "=O" is now O, terminal, Double.
  - The fragment menus have new names.
  - The named types stay as an API (`edit.atom_type`, short names in the geometry list), but the 19-type list and
    its order are gone.
- **Metals.**
  - `COORDINATION` geometries with `IDEAL_SITES`. Sites are fitted to existing bonds by Kabsch over all assignments
    (at most 720 for an octahedron with five bonds).
  - Metals have no lone pairs and no hydrogens on open sites. They are perceived by degree when a molecule is
    loaded (four neighbours with one pair opposite: square planar).
  - Under UFF, RDKit leaves a six-coordinate metal's hybridization unspecified, and UFF typing then fails.
    `_set_metal_hybridization` sets it from the geometry, and `_uff_fallback` tries the metal's other types when UFF
    has none for the chosen one (square-planar Cu, trigonal-bipyramidal Fe).
  - **The square-planar bug:** UFF's angle parameter for Pt is 90°, but RDKit special-cases only octahedral centres.
    The general cosine term for θ₀ = 90° is K/4 (1 + cos 2θ), whose maximum is at 180°, so trans ligands sat on an
    unstable point. A symmetric NiCl4 stayed put, while cis-PtCl2(NH3)2 folded to 49-108°. Trans pairs are now held
    at 179-180° (`UFFAddAngleConstraint`), and the result is 90° and 179°. Test: mixed-ligand Pt, with and without
    the relax.
- **UI-1b is not a bug in the app.** The stale tab bars showed only when a script pumps `processEvents()` without
  `DeferredDelete`; a real event loop shows none. Hiding the bars by hand crashed Qt (gdb: `QWidget::ensurePolished`
  under `QMainWindowLayout::animationFinished`, reached from a deleted widget's animation), so that fix was dropped
  and the smoke script now pumps events as the tests do.
- **The declutter test was a real race.** `ResultsFolder._record` runs from `singleShot(0)` and stored whatever height
  the dock had then. On a slow loop, the test's resize to 360 px came first, so unfolding restored 190 px over the
  user's height. A traced run showed `[record] height=360 folded_to=360`, then `[unfold]`. It now records only a
  height below the one before the fold.
- **Shortcuts.** `gui/action_ids.py` names the actions (`view.mode_*`, `edit.undo`, `edit.redo`, `build.model_kit`)
  and holds their default keys. Undo and Redo are the platform's standard keys (plus Ctrl+Y). None of the keys were
  chosen to match another program; Ctrl+4 continues QuantaBench's Ctrl+1 to Ctrl+3 mouse modes.
- **Merged after UI-4** (#39): Edit > Undo / Redo now come before UI-4's Paste, Copy molecules and Copy as, so
  `test_molecule_list_context_menu_and_edit_menu` expects them first.
- **macOS CI after that merge:** `test_pipeline.py`'s tests with a stand-in gfn1-xtb failed twice at the memory guard
  (2.4 GB in use against a 2.4 GB budget, the runner's free memory: more GUI tests run before them in one process).
  They test conformer failures and the CPU fallback, not the guard, so they now run with it off (`NO_GUARD`).

## 2026-11-22: UI-9 with UI-4 and UI-7 merged (branch `feat/ui-9-controls`, PR #50)
- After #39 (UI-4) and #44 (UI-7) merged, `main` was merged into the branch and their actions moved onto the keymap:
  `clipboard._named_action` binds through `keymap.bind` (Copy molecules in the Molecules dock's scope; `DEFAULT_KEYS`
  removed, `default_shortcuts` reads `SPECS`, which gained the canonical-SMILES Copy as); UI-7's `gui/action_ids.py` is
  gone (its names were the registry's already); Build > Minimize is the action `kit.minimize` and the Minimize
  button's tooltip shows its key. No literal shortcut is left outside `keymap.py`.
- A crash found on the way: with the Model Kit's dock, `test_gui_workspaces` crashed in Qt after other GUI modules
  (gdb: a dock tab bar retired with deleteLater was destroyed under its running move animation, whose end shows the
  half-deleted widget). `WorkspaceManager` now restores and applies layouts with `QMainWindow` animations off.
- Tests: cloud (xvfb) the GUI files, viewers and `test_edit`: 317 passed. Windows full suite (offscreen Qt,
  `PYVISTA_OFF_SCREEN=true`): 1,899 passed, 23 failed, all TEST-WIN-1: 14 `test_lanes`, 2 real-MOPAC, 3 GUI (also on
  `main`), and the memory guard late in a full run (`test_vibrations` x3, `test_pipeline::test_gfn1_xtb_real`; alone
  they pass, 23 of 23).

## 2026-11-22: the lifeline guard for native programs on macOS and Linux (ENG-3b, branch `feat/eng-3b-posix-lifeline`)
- **Why:** Brandon asked for the same watchdog on the MacBook. Until now a GUI or CLI killed outright there (SIGKILL,
  a crash) left its xtb, CREST, MOPAC or Psi4 running until they ended; only cancels, time limits and dead lanes were
  covered.
- **How:** `run_program` starts every native POSIX program as `/bin/sh -c GUARD /bin/sh -c STACK_WRAPPER prog args`
  in its own session, with stdin a pipe it holds open: the guard `wsl:` programs got in ENG-3 (`WSL_GUARD` is now an
  alias of `GUARD`). When the process holding the pipe dies, the kernel closes it, the watchdog reads EOF and kills the
  program's process group (SIGTERM, SIGKILL 2 s later).
- **What changes for the caller:** `proc.pid` is the guard's (the group's leader, so cancels and the lane registry's
  `killpg` work as before). The guard ends as the program did: its exit code, or the same signal re-raised (`-9`,
  `-11` still reach `ProgramRun.returncode`). The program's stdin is /dev/null, as before. The usage meter samples the
  largest resident set among the guard's descendants (the program and what it starts, `ru_maxrss`'s meaning); the CPU
  time comes through the guard, which waits for the program. A program given `stdin_text` runs without the guard.
- **Tests:** the hard-kill test (a stand-in GUI SIGKILLed, its engine and grandchild end) now runs on POSIX too, and
  fails with the guard switched off; exit codes and signals; the meter's sampled peak (a larger child first, so
  `ru_maxrss` cannot set it). The fake CREST in `test_telemetry.py` killed `os.getppid()` to stand in for a dying lane;
  under the guard its parent is a shell, so it walks up to the lane. On the MacBook (a temporary clone): the targeted
  tests pass (the load sampler test fails there on `main` too, under load), and a real xtb MD run and a real CREST
  search in a child process killed with SIGKILL ended 2 s later, guard and all.

## 2026-11-22: SETTINGS-1 follow-ups (branch `feat/settings-1-followups`)
Brandon's answers on PRs #40, #45 and #46: keep a salt's counter-ion as a separate entry; cap Automatic lanes at the
physical cores; check whether Thorough ring conformers should seed from more than 5 conformers.
- **Salts** (`chem.salts`, `BuildConfig.counterion_entries`):
  - The input list is expanded once, before `jobs.run_resumable` takes its chunk and in `pipeline.run`, so the
    entries depend only on the list and the settings: every machine of a grid makes the same ones, and a resume with
    the same list adds none.
  - Counter-ions are the fragments the build removes (`standardize(info=...)`, after the build filters) that are
    ions or named counter-ions. Water, a copy of the kept ion and neutral molecules are left out. HCl counts, as
    chloride.
  - Entries are de-duplicated by the InChIKey of the fragment as drawn, also against the list's own single-fragment
    inputs.
  - Each entry's id is `<first record>_<formula>`, cut to the worker's 60 characters. Its `counterion_of` lists every
    record it came from (`;`), and it copies the posted fields of the first record.
  - The GUI gives an entry a row when its result arrives.
  - No new counter-ion names ("keep what we have for now").
- **Lanes:** `plan_auto` already gave one lane per physical core for most engines. The 2-thread lanes of external
  programs could exceed the physical cores on a machine with more than two threads per core; now every automatic plan
  stops at the physical cores. `parse_lanes` already refused more lanes than logical threads; `plan_text` now
  adds a note when a typed spec runs more lanes than physical cores. The coordinator relayed NOVA's MD and CREST
  measurements: flat scaling past ~6-8 threads per job.
- **Twist-boat seeds**:
  - **What the sources say.** Spartan's manual: Thorough "considers twist-boat conformers of six-membered rings (in
    addition to chair conformers)". Wavefunction's conformer FAQ: twist-boats "are unlikely by default", and its
    FINDBOATS keyword samples them. Wang, Witek, Landrum and Riniker (J. Chem. Inf. Model. 2020) add small-ring torsion
    potentials to ETKDG. None says how many conformers to start twist-boats from.
  - **The check.** 45 molecules with a saturated six-membered ring, sampled from the QSAR table (seed 7; 38 with one
    such ring, 7 with 2-4; at most 40 heavy atoms). Each ran gas MMFF94s (normal, thorough with 5 seeds, thorough with
    20), then AIMNet2 on the 20 lowest of each thorough pool, ranked by the refinement. On the workstation, 12
    processes at below-normal priority, about 45 min.
  - **Results:**
    - 20 seeds gave a larger force-field pool for 30 of 45 molecules, with new conformers among the 20 lowest for 27.
    - The lowest force-field energy dropped for 6, by 0.07-4.85 kcal/mol.
    - After the refinement the best conformer changed for 6 of 45, always to a lower one: 0.11-2.83 kcal/mol lower,
      heavy-atom RMSD 0.47-2.59 Å. These were mestranol, progesterone, medroxyprogesterone, a hydroxy-keto
      steroid (53-05-4), hydroquinine and cinchonamine: all fused polycyclics. No monocyclic molecule changed.
    - Cost per molecule: force field median 6.3 s with 5 seeds against 15.6 s with 20 (2.6x in total), refinement
      +20 %.
  - **Decision.** The default is now 20 (`TWIST_BOAT_SEEDS`, `twist_boat_seeds`, Build defaults > *Twist-boat
    seeds*, `--twist-boat-seeds`), as an advanced science setting.
    - It matters only with Thorough (`_ONLY_WITH`).
    - A Thorough run stored before it ran with 5 (`LEGACY_DEFAULTS`).
    - Not tested: more than 20.
- Tests: `tests/test_salts.py`, plus additions to `test_build_filters.py`, `test_gui_settings_build.py` and
  `test_lanes.py`.

## 2026-11-22: lane plans fit the free memory (MEM-1, branch `fix/mem-1-lane-plan-free-memory`)
- **Why:** the workstation ran short of memory during batch runs. Its hardware monitor's log (one sample a minute,
  2026-11-08 to 11-22) shows the lowest free memory, under 1 GB, during the Psi4 tranche batches of 11-12/13 (five 4 GB
  Psi4 lanes). Other programs use about 20 of its 31.2 GB, while the lane plan sized the lanes on 0.75 x 31.2 GB = 23 GB.
- **The change:** `lanes.plan_auto` caps the lanes by `min(0.75 x RAM, available - 2 GB)` (`lane_memory_gb`; the
  available memory at the plan is `HardwareReport.ram_available_gb` and `cpu_report()`'s). With 11 GB free, that batch
  plans one 4 GB Psi4 lane, not five. A report without the field plans as before. The reason says
  `capped by memory (11 GB free of 31 GB)`.
- **WSL** (measured on the workstation): the VM's default cap is half the RAM (15.2 GB there), with 4 GB of swap, and
  it shuts down when idle (`vmIdleTimeout`, 60 s). A GFN-FF CREST search through WSL took about 0.05 GB of WSL's
  available memory and under 100 MB of /dev/shm. Microsoft documents `autoMemoryReclaim` (`[experimental]`, default
  `dropCache`). /dev/shm is not cache: it is freed when QuantaBench removes the job folder.

## 2026-11-22: FIX-8, a cancel reaches a paused job (branch `fix/fix-8-cancel-paused`)
- **Found** when CREST-RUN-1's water phase was restarted on the workstation alone: `grid cancel` on the old grid
  printed "job d348feda1d4b paused" for the MacBook's share and "cancelled" for the other one. `JobManager.cancel`
  changed queued and interrupted jobs to cancelled and asked running ones to stop; a paused job (DIST-7: its batch
  stopped by a pause with drain, molecules left) matched neither branch, so the call only logged "cancel requested".
  `set_paused(False)` requeues every paused job, and the MacBook's pause was its time window's, which lifts itself at
  02:00: the old share would have started again there.
- **Fix:** `cancel` treats "paused" like "queued" (state, finish time, `cancel_requested`, inputs dropped, job list
  saved), then, outside the lock, `_finish_run_folder`: the job's batch is not running, and its worker.json still said
  "paused" from the end of that batch. That is the same step a job cancelled while the worker was down gets at start.
- **Test:** `test_a_paused_job_can_be_cancelled` (stand-in molecules of 0.25 s, a real worker over HTTP): pause with
  drain, the job becomes paused, cancel returns "cancelled" with a finish time, `paused_jobs` is empty, the run
  folder's worker.json and the saved job list say "cancelled", and after a resume nothing starts in a second. Without
  the fix it fails at the first check. `test_machine_control.py`, `test_worker.py` and `test_work_sharing.py`: 74
  passed (Linux).

## 2026-11-23: long-range settings for slabs (MD-LR, branch `feat/md-lr-slab-defaults`)
- **The ask** (Brandon, 2026-11-22): options for the long-range method, the grid and the "screening length" (the LJ
  cutoff), and recommendations based on the setup. Background read first: the pilot A/B/C reports and summary, the
  mesh check (2026-11-22), A-fine's README and logs, the split-test plan and the literature review (2026-11-22).
- **Slab-safe pppm/disp, by rule rather than by accuracy.** Two ways were possible: A-fine's accuracies
  (`force/disp/real 1e-5`, `force/disp/kspace 2e-5`) or an explicit G and mesh. The explicit rule won: LAMMPS'
  estimate assumes a uniform density (a slab's vacuum dilutes it, which is how A got 2 x 2 x 8), it gave A-fine a
  different G in the bulk (0.2375) and the slab (0.2305), and a rule can be checked before a run. G·rc = 2.77 and
  h·G = 0.52 are A-fine's slab values; with A's box they give A-fine's 18 x 18 x 90 exactly. The mesh is computed in
  the input (`mesh/disp $(ceil(lx/v_disp_h):%.0f) ...`) after `read_data` and again after `change_box`, so it follows
  the real box (lattice or Packmol start, NPT, `-var vacuum`). Checked with the PyPI LAMMPS (22 Jul 2025): pppm/disp
  accepts `gewald/disp` + `mesh/disp` without the two accuracies, rounds a requested 53 up to 54 (products of 2, 3, 5;
  `disp_mesh` does the same), re-reads the mesh after the box opens (hexane smoke slab: 12 x 12 x 9 in the bulk box,
  12 x 12 x 54 after), and `in.run` (bulk, `clear`, slab) runs with the index variables defined in both.
- **Recommendations from the pilots, checked against the reports:** GPU → 16 Å + tail (B: 11.71 ± 0.30 vs Tsige &
  Grest 11.25; 21.0 ns/day on GPU + 6 threads); CPU only → converged pppm/disp (6.6 ns/day on 8 threads in the mesh
  check's benchmark; 16 Å on the CPU ran at 3.9 on 4 threads); reference values → converged pppm/disp; a warning for
  12 Å slabs (C: plateau −4.7 %). Bulk → 12 Å with LAMMPS' homogeneous tail: A-fine's finished 1 ns NPT averages
  1.6720 ± 0.0013 g/cm³ (last 0.8 ns, 10 blocks, computed here from its `bulk_density.txt`), the same as B's 1.6719
  and C's 1.6714 with the tail, so A's 1.6645 was the coarse mesh.
- **Two numbers in the brief that the reports don't back:** B's GPU + CPU speed is 21.0 ns/day in its report (20.98),
  not 24; and the CPU comparison (6.6 vs 3.9 ns/day) is 8 threads against 4. The table uses the reports' numbers
  and says the thread counts.
- **The CLI's default is now the recommendation** (`--long-range auto`). On a machine whose LAMMPS has the GPU
  package (the workstation's installer `lmp.exe`), a plain `md liquid-slab` writes a 16 Å cutoff folder; on the Mac
  or in CI, pppm/disp. `--gpu yes|no` states the machine that will run the folder when it is not this one. The
  Python default (`LiquidSlabConfig()`) stays `ewald` with the slab-safe settings.
- Nothing on the workstation or the Mac was touched (A-fine, the GAFF2 slab and the GPU check kept running); A-fine's
  logs and `bulk_density.txt` were only read.

## 2026-11-23: a small slab MD benchmark (MD-BENCH-2, branch `feat/md-bench-2-small-slab`)
- **The ask** (Brandon, 2026-11-23): a benchmark for the MD section, with a smaller simulation. After MD-LR (#52) the
  cost of `ewald` depends on the slab box (pppm/disp's mesh follows the box, vacuum and all), so the bulk cubes of
  MD-BENCH-1 no longer time what a σ run does.
- **The box.** 190 C6F14 (3,800 atoms) in the pilot's 40 x 40 Å lateral box, ~39 Å of liquid and half that of vacuum
  on each side. Why those proportions: every `md liquid-slab` size has box / liquid = 2 (40 or 50 Å lateral, 100 Å
  of liquid, 2 x 50 Å of vacuum), so with the same ratio the dispersion mesh per atom is the pilot's (18 x 18 x 36
  for 3,800 atoms against 18 x 18 x 90 for 9,780: 3.07 against 2.98 points per atom) and ns/day scales by the atoms.
  50 Å of vacuum on a 39 Å slab would have given 1.8 x the pilot's mesh per atom. The lateral edge keeps 16 Å
  cutoffs legal (> 33 Å). What it does not match: a 39 Å slab has more of its atoms near a surface than a 100 Å one,
  so its pair work per atom is somewhat lower; `--md-size pilot-slab` is there to measure that once.
- **The slab's records** carry the k-space grids LAMMPS reported and its timing breakdown. Cloud smoke run (PyPI LAMMPS
  22 Jul 2025, 1 thread, 30 s per variant): cpu-ewald 3.96, cpu-cutoff12 5.57, cpu-cutoff16 3.07 ns/day; Coulomb grid
  24 x 24 x 48 and dispersion 18 x 18 x 36 for `ewald` (k-space 29 % of the time), 16 x 16 x 25 for `cutoff12`
  (12 %); box built in 100 s, 207 s in all. Hence `scaled_rate`: the k-space share scales with the box volume when
  the job's box differs in shape (a thinner slab in 2 x 50 Å of vacuum), the rest with the atoms.
- **The MacBook's failure (2026-11-21), what the records show.** Its registry file names the LAMMPS source as "this
  environment (environment variable: C:Users<name>/miniforge3/envs/qb-lammps/bin/lmp not found)": the variable was
  set in a command sent from PowerShell on the workstation, where `$HOME` expanded to the Windows home folder, and the
  Mac's shell dropped the unquoted backslashes. No LAMMPS path for the Mac was in the settings. `find_program` skipped
  the bad value and took the next hit, the venv's PyPI wheel (`.venv/bin/lmp`, a `#!/bin/sh` launcher for a Python
  script that runs the real binary), whose macOS build has "OPENMP package API: Serial": the pilot box's set-up ran on
  one thread. Reproduced here: with such a value in `QUANTABENCH_LAMMPS`, main's `find_lammps` returns the venv's
  `lmp` with the same source string; the branch refuses it.
- **The exit 143** came about 2 minutes after the start (the run's output file on the Mac, read only). 143 is how a
  shell reports a child killed by SIGTERM. QuantaBench was not the sender: it stops LAMMPS only on a cancel (the record
  then says `cancelled`) or at its 3,600 s limit (`timed out`). LAMMPS crashes give SIGSEGV/SIGABRT, memory kills
  SIGKILL. The same session wrote the GAFF2 start script 24 s after the benchmark began and, after it died, looked for
  the benchmark's `log.prep` and for time limits in `bench.py`, so the stop was not its plan. Which process sent the
  signal is not known: the benchmark deleted its set-up folder, logs and all. Here the same LAMMPS version (the Linux
  wheel) on one thread builds that pilot box without error in 273 s, and PyPI's wrapper does not explain 143 on
  its own (on Linux a SIGTERM to the wrapper gives -15, to the binary 241, and the binary is orphaned when only the
  wrapper is killed). Now a failed set-up keeps its inputs and logs, and the reason names the signal (`exit_reason`).
- Nothing was run on the workstation or the Mac besides read-only checks (the Mac's run output, its `.venv/bin/lmp`
  launcher, `/bin/sh` → bash, its conda envs) and the Windows test suite.

## 2026-11-23: MD runs as jobs, SIM-1 PR 1 (branch `feat/sim-1-md-run-jobs`)
- **The ask** (Brandon, 2026-11-23): start building the Simulate workspace; later, default geometries and common
  setups. The design's v1 is two PRs; this is the first: segments, pause and resume, live telemetry, the CLI, MD jobs
  in the queue. Read first: SIMULATE_DESIGN.md §5/§7/§12/§14, MD-LR (#52), and the split test plan (portable
  checkpoints are a data file plus the step; binary restarts only for the same machine and build).
- **LAMMPS facts checked with the PyPI build (22 Jul 2025) before building on them:**
  - `fix halt ... error soft` ends the current run but not the input: the later runs are skipped (the timer is set to
    time out) and the input goes on, so a plain `in.bulk` would have written `bulk_final.data` after a skipped NPT.
    Hence a check after every `run ... upto`: a run that stopped short of its end step writes the pause checkpoint and
    quits. A halt that lands exactly on a stage's end is caught by the next stage's (skipped) run.
  - On a resume, `fix ave/time` with one sample per row writes the start step again at its setup, as does a dump;
    `fix ave/chunk` with 100 samples per block does not (its window would start before the step). So the cut before a
    resume removes the step itself for the first two and keeps it for the third; and pauses land on multiples of the
    output periods (`pause_every`, the least common multiple), else a profile block would be lost.
  - The restart header's flag numbering is not what the source I remembered had: 22 Jul 2025 writes VERSION 0,
    SMALLINT 1, IMAGEINT 48, TAGINT 2, BIGINT 3, UNITS 4, NTIMESTEP 5. The first parser (UNITS 5, NTIMESTEP 6)
    returned None for every file, and the supervisor quietly fell back to the data file: caught in the manual
    end-to-end run. `restart_step` now finds the units string by its content and takes the bigint after it.
  - `write_data` puts "timestep = N" in its first line, so a data checkpoint describes itself too.
  - `jump SELF label` inside `if` works for skipping finished stages in a segment input (`-in file`).
- **End-to-end in the cloud** (540-atom hexane slab, 1 thread): a straight run (two segments), a pause in the slab at
  3,000 and a resume (one pressure-tensor series 0-31,000 every 10 steps, profiles every 100, frames every 500, the
  thermo row at 3,000 identical in both logs), a stop during the slab and a resume from the periodic restart at 3,000
  (83 rows cut, then appended), a pause at the compress/NPT boundary in the bulk and its resume through the `jump`s.
- **GUI:** an MD job starts `quantabench-cli md run|resume DIR` detached (Windows: no console, its own process group,
  out of the caller's job where allowed) and follows `run.json` and `progress.json` every 2 s. It runs beside builds;
  one MD job at a time here, as the design's per-resource rule until SCHED-3. A supervisor started this way logs only
  to `supervisor.log` (its stdout is that file, so it does not echo too). Scripted smoke test under xvfb with
  screenshots: running (0.0074 / 0.207 ns · 62.4 ns/day · ETA 5 min), pausing, paused with Resume enabled.
- **Not changed:** the liquid slab's default dispersion (the design said `cutoff16`; MD-LR's recommendation decides
  it now), `md/bench.py` (MD-BENCH-2 is changing it; only its `find_lammps`, `can_run`, `accelerator_args` and
  `default_threads` are used).
- **Docs:** SIMULATE_DESIGN.md (status approved, §5 as built, §7 Brandon's scheduler clarification, §12 PR 1 as built,
  §14 the recommendations PR 1 takes, §15 setup templates and pre-flight checks), ENGINES.md ("MD runs as jobs"),
  ARCHITECTURE.md, UI_PLAN.md; BACKLOG SIM-1, new SIM-2 and SIM-3, SCHED-3.
- **Windows (the workstation, the installer's `lmp.exe` 4 Jul 2026 on PATH):** the full suite on 38b8953: 19 failed,
  1945 passed, 95 skipped, 2 xfailed, 2 xpassed in 20 min 37 s. The 19 are the ones MD-BENCH-2's run on the same
  machine had this night (14 in `test_lanes.py`, 2 MOPAC, 2 in `test_gui.py`, `test_gui_workspaces.py`'s switching
  test; TEST-WIN-1), none in SIM-1's files. SIM-1's tests with `test_md_liquid.py` and `test_lammps.py`: 66 passed,
  3 skipped (AmberTools; the GPU test), in 2 min 48 s; the supervisor tests ran with the real `lmp.exe`, and the GUI
  test started a detached supervisor from the offscreen window. With `QB_TEST_MD_GPU=1` the CPU-then-GPU test passed
  on the RX 9060 XT (OpenCL, `device_type generic`, 2 threads, below normal, 32 s): one series across the join, the
  potential energy at the join within 1e-4.
- Nothing on the workstation or the Mac was started or stopped (A-fine, CREST's water phase and the GAFF2 slab kept
  running). The tests above ran 540-atom LAMMPS jobs there on 1-2 threads at below-normal priority, behind the
  pytest lock.

## 2026-11-23: OpenMM as a second MD engine, with the Apple GPU (OPENMM-1, branch `feat/openmm-1-openmm-engine`)

Brandon approved keeping both engines and writing the Apple-GPU code for OpenMM, with a note in the engine choice
("OpenMM runs much faster at reduced accuracy"). Built on SIM-1's supervisor (#55) and MD-BENCH-2 (#54), from the
MAC-GPU report's prototype converter (`Runs\mac-gpu-2026-11-23\scripts\bench_c6f14.py`).

- **Converter** (`md.openmm_engine`). Parity with LAMMPS `run 0` on minimized boxes (`tests/test_md_openmm.py`,
  run here with LAMMPS 22 Jul 2025 and OpenMM 8.6.1 from PyPI): C6F14 OPLS-AA at 12 Å (slab, bulk + tail, LJ-PME),
  C6F6 (cvff impropers), GAFF2-like C6F6 and C6F14 (fourier, LB, 1-4 Coulomb 1/1.2; bulk + tail, LJ-PME) and an
  explicit `pair_coeff 1 2`: bonded terms to 1e-6, nonbonded within 2e-4 of its terms, all on the first try. On pilot
  B's slab OpenMM gives 497.80 kcal/mol; LAMMPS 497.40 at its PPPM 1e-5, 497.76 at 1e-6, 497.79 at 1e-7: the
  prototype's 0.4 kcal/mol "mismatch" was LAMMPS' k-space error.
- **σ: what the frames can and cannot do.** Thinning pilot B's own 10-fs pressure series: block error 0.30 mN/m every
  10 or 20 fs, 0.38 every 50 fs, 0.81 every 100 fs, 2.9 every 1 ps, 7.3 every 5 ps (10 ns). The virial decorrelates
  in ~20 fs (g = 2.1 at 10 fs; bond vibrations), so 1-ps frames (the MAC-GPU plan) are ten times less precise than
  pilot B, and dense frames are too big (117 KB each). So the run samples σ itself every 20 steps: two energies at
  the area strained by ±ε at constant volume (atoms moved with the box), a central difference that is the
  Kirkwood-Buff virial. Measured cost on pilot B's slab: M3 Max 84 → 67 ns/day, RX 9060 XT 137 → 90 (mixed).
- **Making the on-the-fly σ equal LAMMPS' virial** (small hexane slab at 8 Å, 101 frames, LAMMPS rerun as reference):
  - Without anything: +1.0 mN/m. Split on one frame (Reference platform, ε = 1e-6): the LJ part agreed to 0.0005; the
    Coulomb part was −0.91 against LAMMPS at PPPM 1e-5, but 1e-6 and 1e-7 gave 23.71 and 23.70 and OpenMM 23.69: the
    LAMMPS side was off. On pilot B's frame LAMMPS' σ at 1e-5 is +0.36 above 1e-6/1e-7 (−239.74 vs −240.09/−240.10),
    and OpenMM's finite difference gives −240.04.
  - A truncated LJ also jumps at rc: pairs crossing it under the strain add an "impulsive" term the force virial does
    not have (+2 mN/m here). A force group that is never integrated adds −U_ij(rc) per pair within rc (zero force),
    which cancels the jumps. The LJ force still jumps at rc, so a central difference is biased in proportion to ε:
    −0.16 mN/m at 1e-3, −0.04 at 2e-4, −0.03 ± 0.04 at 5e-5 (Reference platform: −0.15/−0.03/−0.02). Default ε = 1e-4.
  - So the rerun uses Coulomb k-space at least 1e-6, and the check in `md surface-tension` compares like with like.
    The real-run test (540-atom hexane on OpenMM's CPU platform, paused and resumed) passes with |mean difference| <
    0.3 mN/m over its frames.
- **Runner** (`md.openmm_run`): bulk (minimize, warm-up at 0.5 fs with a 50-fs Langevin, NPT with the Monte Carlo
  barostat) and slab (opened in Python like `in.slab`, `slab_start.data` for the rerun), LAMMPS-shaped logs, the same
  output files, state-XML checkpoints (`stepCount` inside), pause at `pause_every` (now also a multiple of the frame,
  σ-sample and test-area intervals), periodic `state.a/b.xml`, cut and resume through SIM-1's supervisor (DCD frames
  cut by `md.dcd.truncate`). Found on the way: `md run --detach` dropped `--priority` (the detached supervisor ran at
  the settings' priority): fixed.
- **The workstation** (RX 9060 XT, OpenCL, below normal, beside A-fine and CREST): OpenMM sees single, mixed and double; pilot B's
  slab 135 (single), 137 (mixed), 121 (LJ-PME 12 Å, mixed) ns/day; with σ samples every 20 fs 97 / 90. The
  `qb-openmm` environment was created there from `envs/openmm.yml` (and on the MacBook).
