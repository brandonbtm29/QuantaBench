# Backlog

Ordered by priority. Nightly runs take the first `ready` item with no open PR and whose dependencies
are merged (see `RUNBOOK.md`). Status: `ready` · `in-PR` · `done` · `blocked` · `needs-Brandon`
(hardware, Windows, or a decision: never picked by nightly runs).

Design reference: `docs/UI_PLAN.md`. Keep QuantaBench its own app: borrow features, not layouts.

---

## UI-1: App shell with task workspaces · ready
Depends on: none (everything else in the UI plugs into this, so it goes first).
- Left activity bar switching workspaces: **Build, Calculate, Simulate, Analyze, Library**. Each is a
  page/dock set. Map the existing panels: molecule list + inputs → Build; pipeline stages → Calculate;
  surface builder / LAMMPS export → Simulate (stub page with what exists); conformer table, descriptors
  → Analyze; run folders/jobs → Library (stub).
- One **Appearance** panel (style, labels, color mode, hydrogens, show 2D) replacing the scattered view
  toolbar widgets; keep keyboard shortcuts.
- **Playback bar** under the viewport stepping through conformers now (frames/molecules later): first /
  prev / play / next / last + slider + label.
- **Status bar**: molecule count, current molecule/conformer, CPU threads, GPU name and, when on GPU,
  allocated VRAM (refresh ≤ 1 Hz, off the GUI thread if it costs anything).
- Persist workspace/dock layout per machine (QSettings), with "Reset layout" in View.
- Acceptance: all existing features reachable, existing tests pass, new tests for workspace switching
  and playback stepping, screenshots of each workspace committed in the PR.

## UI-2: Project document model · ready
Depends on: UI-1.
- A project (`*.qbproj` folder: `project.json` + the existing per-run `jobs.sqlite` + result folders)
  holding many molecules, their inputs, results and the build settings used. New / Open / Save /
  Save As / recent projects. Unsaved-changes prompt.
- Opening a project restores the molecule list and built results (reads `summary.json` +
  `_conformers.sdf`); "Resume" continues unfinished molecules via `jobs.run_resumable`.
- The playback bar can step through molecules as well as conformers.
- Acceptance: round-trip save/open test with built results; resume of a half-finished project;
  Windows-safe paths (no absolute paths stored for files inside the project).

## UI-3: Spreadsheet, plots and formula columns · ready
Depends on: UI-1 (UI-2 nice to have).
- Analyze workspace: sortable/filterable table of molecules × properties (descriptors, energies,
  thermo), column chooser, copy/paste to Excel, CSV/XLSX export; selecting a row selects the molecule.
- Plot panel (pyqtgraph or matplotlib QtAgg): scatter/line of any two columns, color by a third,
  point click → select molecule, linear fit with R².
- **Formula columns**: safe expression evaluator (no `eval` of arbitrary Python; e.g. `asteval` or a
  small AST whitelist) over column names, plus a "fit coefficients" helper that fits parameters
  (scipy least squares) to a target column, e.g. the quantum parachor
  `γ = ((a·α + b·μ + c) / V)^4`. Saved with the project.
- Acceptance: tests for the evaluator (rejects attribute access/imports), fitting recovers known
  coefficients on synthetic data, GUI smoke screenshot.

## UI-4: Structure editor with PFAS fragment palette and undo/redo · ready
Depends on: UI-1.
- Build workspace: 2D editing on the existing RDKit 2D view: change element, add/delete atom,
  bond order cycle, formal charge ±, make/break bond, clean 2D; live 3D update with quick MMFF minimize.
- **Fragment palette**: CF3, (CF2)n chain builder, ether O, carboxylate/carboxylic acid, sulfonate,
  sulfonamide, fluorotelomer CH2CH2 linker, phosphate, common rings; attach at the selected atom.
- **Undo/redo** (QUndoStack) for all edits, Ctrl+Z / Ctrl+Y.
- Acceptance: tests that build PFOA, PFOS, GenX and 6:2 FTS from fragments and match canonical SMILES;
  undo/redo round-trips; screenshot.

## UI-5: Job dialog (level of theory × task × resources) and job monitor · ready
Depends on: UI-1.
- Calculate workspace: one dialog to submit selected molecules. **Level of theory** grouped by family
  (force field / tight binding / ML potential / DFT (disabled until available)), each with charge,
  multiplicity, solvent (where supported); **task** (single point, optimize, conformer distribution,
  frequencies + thermo, scan (stub)); **resources** (threads, GPU on/off/device, memory cap; leave room
  for "Run on: machine", DIST-3).
- **Recipes**: named multi-level protocols saved in settings (e.g. "Screen: MMFF conformers → GFN2 →
  AIMNet2"), mapped onto `BuildConfig`. Ship 3 sensible presets.
- **Job monitor** dock: queued/running/done jobs, per-job progress and log, cancel, open output folder,
  CPU threads and GPU memory in use.
- Acceptance: tests that each preset produces the right `BuildConfig`; queue runs jobs in order with
  cancellation; screenshot.

## RES-1: MD front-end survey and Simulate workspace design · ready
Depends on: none (research only; can run any night).
- Survey how LAMMPS-GUI, VMD + QwikMD, CHARMM-GUI, OpenMM-Setup, Materials Studio Forcite/Amorphous
  Cell, MAPS, SAMSON, Packmol/moltemplate and OVITO handle: system building, force-field assignment,
  protocol setup, run control, live monitoring, trajectory analysis.
- Deliver `docs/SIMULATE_DESIGN.md`: proposed Simulate workspace flow, data model (System, ForceField,
  Protocol, Run), engines (OpenMM, LAMMPS), what to build in which order, open questions for Brandon.
  No code. Cite sources.

## SIM-1: Simulate workspace v1 · blocked (needs RES-1 design approved by Brandon)

## FIX-1: Windows CI green · ready
- Make sure `.github/workflows/ci.yml` passes on `windows-latest` and `ubuntu-latest` with core deps
  (check the first CI runs after 0.2 is pushed; Mesa OpenGL on Windows).

---

## WIN-1 (see FORK-1): xTB dipole integrals on native Windows · done 2026-11-05 (FORK-1)
- Native Windows has no compatible dxtb `libcint` backend (tad-libcint 0.4 needs tad-mctc ≥ 0.11, dxtb 0.4
  pins 0.7), so GFN2-xTB, xTB polarizability and xTB IR intensities are unavailable there (they degrade
  with a note). Investigate: a newer dxtb release, building tad-libcint 0.3 for Windows (CI wheel), or
  a charge-response/finite-field fallback. Acceptance: GFN2 and IR intensities on Windows, or a
  documented decision why not.
- 2026-11-01: GFN2-xTB is now available on Windows through the xtb program (`xtb-gfn2`, finite-difference
  IR intensities from its dipoles). What remains is dxtb's GPU/batched GFN2 path; lower priority.
- 2026-11-05: done through FORK-1: the dxtb fork with tad-libcint 0.4 gives GFN2, polarizability and IR
  intensities natively on Windows (NOVA: matches Linux to ~1e-15 to 4e-9).

## MD-1: Counter-ions in vacuum SMD (Windows LAMMPS non-determinism + ion-pair heating) · ready
- On BTM NOVA, PyPI's Windows LAMMPS wheel (2025.7.22.4) fails ~5-10% of *identical* runs of a
  TFA⁻ + Na⁺ on graphite setup with "Out of range atoms - cannot compute PPPM" (also with
  `OMP_NUM_THREADS=1`: 1/20); Linux is deterministic and passes 12/12 seeds. Reproducer:
  `scratch/ionrepro.py` (in git history of docs/dev; regenerate from tests/test_lammps.py setup).
- Physics side: a free Na⁺ falls onto the carboxylate in vacuum and releases tens of kcal/mol; even with
  Joung-Cheatham ion LJ and 20 fs ion friction the adsorbate runs at 1000+ K in short protocols.
  Options: start the counter-ion pre-paired at the head group, longer soft start, freeze spectator ions,
  or recommend no counter-ion (PPPM neutralizing background) for adsorption SMD in vacuum.
- Acceptance: a documented protocol that keeps T within ±20% of target after the soft start, and a
  minimal upstream LAMMPS issue (or a confirmed local cause) for the Windows failures.
- 2026-11-01: not only counter-ions: `test_smd_run_with_lammps_module` (TFA⁻ alone on graphite, PPPM)
  failed 2 of 6 identical runs on BTM NOVA with the same error; now also marked flaky on Windows.
  Compare with `lmp.exe` from the LAMMPS Windows installer to tell a wheel problem from an input problem.
- 2026-11-05: a third test fails the same way on NOVA, 1 of 6 identical runs: the LAMMPS run of the periodic-boundary test (TFA alone, 2-layer slab, now `test_molecule_across_periodic_boundary_runs`); marked flaky on Windows too.

## GPU-1 (= FORK-3): AIMNet2 on AMD and Apple GPUs (replace NVIDIA Warp D3) · done on ROCm 2026-11-05; MPS needs-Brandon
- 2026-11-04: Warp is CPU-only on macOS too, so the same adapter enables AIMNet2 on the MacBook's GPU (MPS,
  float32). Test the adapter on `mps` as well (needs-Brandon), and offer the change upstream to `aimnet`.
- `aimnet` 0.2 computes DFT-D3 (and neighbor lists) through `nvalchemi-toolkit-ops` / NVIDIA Warp, which
  rejects ROCm devices ("Invalid device identifier: cuda:0" on BTM NOVA). The NN itself is plain PyTorch.
- Load the model with dispersion disabled (`needs_dispersion=False`) and add D3(BJ) for ωB97M from
  `tad-dftd3` (pure PyTorch, already installed with dxtb) with AIMNet2's D3 parameters; keep the dense
  neighbor path (< `nb_threshold` atoms). Verify on CPU against the stock calculator to ~1e-5 eV and
  forces, then on GPU (needs-Brandon to run). Acceptance: AIMNet2 single + batched on a ROCm device.
- 2026-11-05: `chem/aimnet_torch.py` (Warp-free calculator: dense neighbor path, tad-dftd3 D3(BJ) with
  aimnet's parameters and reference tables). NOVA, real weights: CPU < 3.3e-6 eV and RX 9060 XT < 7.5e-6 eV
  from the stock calculator over 6 PFAS x 4 conformers; batched refinement of 8 PFOA conformers 5.7 s on
  the GPU vs 59.8 s on the CPU. Picked automatically on ROCm and MPS (`QUANTABENCH_AIMNET_BACKEND`).
  Found and fixed a stock aimnet bug on the way (`guard_neighbor_lists`, see DEVLOG). Left: a run on the
  Mac's MPS, and the upstream PR making Warp optional (text drafted; post only with Brandon's OK).

## GPU-2 (= FORK-4): dxtb on AMD and Apple GPUs · done 2026-11-05 (runs, but slower than the CPU)
- 2026-11-04: MPS has no float64 and unverified linalg coverage: the device-split wrapper (dense
  eigh/cholesky on the CPU) must cover `mps` too; then propose a configurable linalg device upstream.
- AMD's Windows ROCm PyTorch has no MAGMA/hipSOLVER linear algebra on the GPU
  (`torch.linalg.cholesky` / `eigh` raise). Options: route dxtb's small dense linear algebra to the CPU
  inside an otherwise-GPU calculation (tensors are ~100x100 for PFAS-size molecules), or document
  WSL2/Linux ROCm wheels (which include MAGMA). Measure whether GPU dxtb beats CPU at all for these sizes;
  batched conformers are where it should.
- 2026-11-05: `chem/torch_compat.py` routes Cholesky/solve (and everything on MPS) to the CPU inside dxtb
  calls; GFN1 on the RX 9060 XT matches CPU float64 to 8.6e-6 Eh. Measured 5-8x *slower* than NOVA's CPU at
  every size and batch size (host synchronizations in dxtb 0.4's overlap integrals, not the routed ops), so
  `dxtb_device()` keeps dxtb on the CPU there by default (Settings > Compute > dxtb on the GPU,
  `QUANTABENCH_DXTB_GPU`). Follow-up (later): profile/upstream the overlap-integral syncs in dxtb.

## PERF-1: Separate CPU and GPU environments on Windows · ready
- 2026-11-05: `install_windows.ps1 -Gpu rocm` creates `.venv-rocm` next to the CPU `.venv`. With GPU-1 done,
  AIMNet2 is ~10x faster on the GPU, while dxtb and MMFF want the fast CPU BLAS: measure a full PFAS batch in
  both environments (lanes `gpu:1x4,cpu:...` in `.venv-rocm` vs `auto` in `.venv`) and pick the default.
- Until GPU-1/GPU-2 land, the PyPI CPU torch (fast MKL/OpenBLAS kernels) is the better default on Windows.
  Document a two-venv setup (`.venv` CPU, `.venv-rocm` GPU) in INSTALL.md and make `doctor --bench`
  report CPU GFLOP/s with a warning when it is far below expectations. Also find why `doctor --bench`
  ran > 50 min under the ROCm build (likely BLAS thread oversubscription: 23 OpenMP threads × OpenBLAS).

## ENG-1: External engines on native Windows · needs-Brandon
- Install `envs/tools-windows.yml` and `envs/psi4.yml` on BTM NOVA, run `quantabench-cli doctor`, then
  `pytest -m external` with `QUANTABENCH_XTB`, `QUANTABENCH_MOPAC`, `QUANTABENCH_PSI4_PYTHON` set (or the
  paths in Settings > Engines). Check: conda's `Library\bin` discovery, process-tree cancellation
  (`taskkill /T`), paths with spaces ("GitHub Projects"), Psi4's PCMSolver on win-64, GUI Engines tab.

## ENG-2: CREST through WSL · needs-Brandon
- `wsl:/home/<user>/miniforge3/envs/qb-tools/bin/crest` from the Windows GUI on a PFAS: `--cd` path
  translation, `WSLENV` thread variables, cancellation of the WSL process, and gfn2//gfnff (needs `xtb` on
  WSL's default PATH). Only covered by mocked tests so far (a fake `wsl.exe` on Linux runs the real wrapper).
- The wrapper (`external.WSL_WRAPPER`) raises `ulimit -s` and writes the Linux PID to `.qb_pid`; cancel
  kills that PID through `wsl.exe` before `taskkill`. Children of CREST's legacy mode (xtb) may survive a
  cancel: if they do, start the program with `setsid -w` and kill the process group.

## ENG-3: Engine follow-ups · ready
- `vibrations.from_hessian(...)`: make the second half of `analyze()` public so `qm.vibrations_from_hessian`
  stops duplicating it (see the QM builder's suggested signature in DEVLOG 2026-11-01).
- QM charges (MBIS/Löwdin) as an option for the exported conformer charges (extxyz, LAMMPS data).
- Surfaces: a color bar in the 3D view; split A₊/A₋ at the V = 0 line instead of per vertex.
- `doctor`: probe the Psi4 environment for the pint/qcelemental break and missing dftd3/dftd4.
- tblite on Windows: document or support running QuantaBench inside the conda env that has tblite-python.
- GBSA as a user choice (factories accept `solvation_model`; BuildConfig always uses ALPB). The xtb program
  stops with "not parametrized" for GBSA ethanol/octanol/ethyl acetate (GFN1 also DMF, hexane): fall back to
  ALPB with a note, as tblite does.
- GIPF statistics at 0.001 au as well (the Murray-Politzer literature convention; `qm_surf_*` is at 0.002 au,
  Spartan's surface, recorded as `qm_surf_iso_au`). Water HF/6-31G*: σ²tot 478 (0.002) vs 345 (0.001).
- Psi4 log: write the MBIS free-atom and finite-field SCFs to a second output file, so cclib can re-import
  `<id>_qm_log.out` (today it stops at the free-atom UKS "Beta Occupied" block).
- Review 2026-11-01 leftovers (robustness): a Windows Job Object (kill-on-close) so a hard GUI crash cannot
  orphan xtb/CREST/Psi4; `subprocess.run(timeout=...)` in version probes can still block on Windows if a
  launcher's child keeps the pipe open (use Popen + kill); the conda scan takes the first env alphabetically,
  not the newest program version; failed-run scratch folders are kept by design but never pruned; the first
  GUI start still probes local program versions on the GUI thread (cached on disk afterwards).
- MOPAC COSMO: energies step as surface segments switch (CF3COO⁻ in water: finite-difference forces off by
  0.3 eV/Å at h = 1 mÅ); consider running MOPAC frequencies in the gas phase automatically.

## VAL-2: ωB97X-D/def2-SVP vs Spartan for the 76 PFAS · ready (after VAL-1 data)
- The QM stage reproduces Spartan's level (Spartan used 6-31G*: run `wb97x-d/6-31g*` too). Compare
  dipoles, polarizabilities, 0.002 au volumes/areas and ESP extrema with Spartan's exports; recommend a
  default level for the 4,700-PFAS descriptor run with timings (PFBA anion: ~4.5 min on 2 threads with
  PCM, cubes and finite-field α).

## Platform and multi-machine track (plan: docs/PLATFORMS.md)

### PLAT-1: First run on the MacBook · needs-Brandon
- INSTALL.md's macOS steps on the MacBook: `doctor --bench`, full suite, `pytest -m external` with
  `envs/tools.yml` + `envs/psi4.yml`, GUI by hand (HiDPI picking, trackpad pinch/rotate, dark mode).
  Record chip, cores and RAM in PLATFORMS.md section 2, and anything that breaks as new items.

### PLAT-2: macOS in CI · in progress
- `macos-14` is in the CI matrix as non-blocking (2026-11-04). Make it green, then required. Add a second
  workflow job: micromamba `envs/tools.yml` + `envs/psi4.yml` on ubuntu-latest and macos-14, then
  `pytest -m external`.
- 2026-11-05: `.github/workflows/external-engines.yml` does the second part (weekly, manual, and on pushes
  touching engine code). Left: watch the first runs and make macos-14 required in `ci.yml`.
- 2026-11-05: first green CI on all three OSes (also fixed: Qt xcb libraries on Ubuntu, `Get-FileHash` under
  pwsh-launched Windows PowerShell, long runner hostnames, Psi4 needs `libxc-c<7`). External engines: Ubuntu
  green; on macos-14 two MOPAC checks fail: the PM7 polarizability is 1.347 Å³ for water (1.259 on Linux,
  from MOPAC's own .out) and 3.22 for methanol (3.0 ± 0.2), while heats of formation, dipoles and orbital
  energies match. Compare the osx-arm64 MOPAC build/version and its POLAR output before relaxing the tests.

### DIST-1: Machine capability report and weighted chunks · done 2026-11-05
- `doctor --json` gains engines, external programs (versions), and `--bench` throughput per engine
  (molecules/hour for a fixed PFAS set). Write it to `machines/<hostname>.json` in the settings folder
  (synced, so every machine sees the others).
- `build --chunk-weights auto` (and the GUI) splits a list in proportion to those throughputs.
- Acceptance: tests for the report schema and the weighted split (deterministic, stable when the list grows).
- 2026-11-05: `dist.profile` (registry in `<settings>/machines/`, `quantabench-cli machines [--refresh]
  [--bench METHODS]`), `dist.plan` (`--chunk-weights 3,1 | name=3,... | auto`), weights stored with the run.
  Same-machine parallelism too: `dist.lanes` (`build --lanes auto | N | gpu:1x4,cpu:5x2`); NOVA: MMFF 3.7x,
  GFN1-xTB 3.45x with lanes.

### DIST-2: `quantabench-cli worker` over Tailscale · done 2026-11-05 (NOVA + MacBook run needs-Brandon)
- Small HTTP/JSON service (stdlib or one light dependency) bound to the Tailscale interface only, with a
  token from settings: submit, status, cancel, list. Each job runs through `jobs.run_resumable` into the
  worker's own run folder under the synced Runs folder (one writer per run folder). Version/commit check;
  environment fingerprint in `summary.json`.
- Windows firewall rule for the Tailscale interface; `caffeinate -i` on macOS while busy; pause switch.
- Acceptance: two workers on one Linux box in tests (localhost), submit/cancel/resume; a manual
  NOVA + MacBook run (needs-Brandon).
- 2026-11-05: `dist.worker` + `dist.client` (`grid machines|build|status|cancel|merge`), docs/WORKERS.md
  (firewall rule, Task Scheduler and LaunchAgent autostart). Left: the two-machine run over Tailscale.

### DIST-3: Machines dock and "Run on" routing · in progress (dock done; Run on needs UI-5)
- Machines dock (status, load, engines). The job dialog's resources get "Run on: this machine / <worker> /
  auto"; auto picks by engine availability and throughput.
- 2026-11-05: minimal Machines dock (registry, worker status on demand, this computer's lane plan). Left:
  the "Run on" choice, which belongs in UI-5's job dialog.

### DIST-5: Speed-aware lanes at the end of a batch · done 2026-11-05 (NOVA measurement needs-Brandon)
- `lanes auto` for AIMNet2 on NOVA gives 1 GPU lane + 4 CPU lanes, and the GPU is ~10x faster per molecule.
  Lanes pull molecules first-come, so near the end of a batch a CPU lane can start a molecule the GPU lane
  would have finished long before; for a small batch the run can end up slower than the GPU lane alone.
  Use each lane's measured time per molecule: when the queue is shorter than (slow lane's time / fast lane's
  time), leave the rest to the fast lanes. Measure on NOVA with 8, 50 and 500 PFAS.
- 2026-11-05: `dist.speed` + `run_parallel`. A free lane leaves the next molecule when the lanes >= 1.5x
  faster, simulated taking the queue in order (each molecule to the one that finishes it first, from when
  each is free), would finish all of it before this lane finished that one. Speeds are seconds per atom per
  (device, threads), measured, else the registry benchmark scaled to the run, else unknown (no holding).
  Only ready lanes (or first starts, <= 60 s) are counted on. On by default (`resources.speed_aware_lanes`,
  `build --[no-]speed-aware-lanes`, Settings > Compute). Simulated (tests/test_speed.py, GPU 5 s and CPU 50 s
  per 20-atom molecule, lanes ready after 4-6 s, benchmark-seeded), makespan with / without / GPU lane alone:
  8 PFAS 46 / 54 / 46 s, 50 PFAS 196 / 204 / 256 s, 500 PFAS 1804 / 1804 / 2506 s. With 10-40-atom molecules
  and 10 % scatter it never ended later than either; with the benchmark ratio 10 % off and 20 % scatter it
  lost up to ~25 % against the GPU lane alone on single 8-molecule runs, never on average.
- Left (needs-Brandon): the real measurement on NOVA. In PowerShell, from the repo with this branch, with a
  PFAS CSV (`pfas.csv`, a SMILES column; add `--smiles-col` if needed) and an empty folder for the runs:
  ```
  quantabench-cli machines --refresh --bench aimnet2          # seeds: aimnet2@cpu and aimnet2@cuda:0
  quantabench-cli lanes auto --method aimnet2                 # expect gpu0 + 4 CPU lanes, speed-aware on
  foreach ($n in 8, 50, 500) {
    Get-Content pfas.csv -TotalCount ($n + 1) | Set-Content "pfas_$n.csv"
    foreach ($run in @(@("on", "--lanes", "auto", "--speed-aware-lanes"),
                       @("off", "--lanes", "auto", "--no-speed-aware-lanes"),
                       @("gpu", "--lanes", "gpu:1x4"))) {
      $t = Measure-Command { quantabench-cli build --file "pfas_$n.csv" --method aimnet2 --out dist5 $run[1..($run.Count - 1)] |
                             Tee-Object -FilePath "dist5_${n}_$($run[0]).log" | Out-Host }
      "$n $($run[0]) $([math]::Round($t.TotalSeconds, 1)) s" | Add-Content dist5_times.txt
    }
  }
  Get-Content dist5_times.txt
  Select-String -Path dist5_*.log -Pattern "Speed-aware lanes|Lane speeds|Molecules per lane"
  ```
  Record the 9 wall times here. If `on` is slower than `gpu` at 8 PFAS, compare the `Lane speeds` line
  (measured) with the benchmark ratio `aimnet2@cuda:0` / `aimnet2@cpu` in `quantabench-cli machines`: the
  8-molecule case relies on that ratio, since the CPU lanes never run a molecule there.

### SCALE-1: The GUI at 14,000 molecules · done 2026-11-05 (batch memory: one follow-up)
- From the PFAS QSAR project (2026-11-05): Spartan could not take 14,000 molecules in one document, so the
  work was cut into 200-molecule batches (1,067 `.spartan` files) that now have to be recombined by hand.
  QuantaBench's engine side already runs any number in one run folder (one job table, one `summary.csv` and
  `descriptors.csv`, per-molecule folders, resumable) and, since 2026-11-05, does not keep the ensembles in
  memory. The GUI still keeps every finished molecule's full result in memory and builds one list item per
  molecule.
- To do: a virtualized molecule list (QListView + model), results kept as summary rows with the full result
  loaded from the molecule's folder on selection (small LRU cache), the conformer table and descriptors from
  `summary.csv`/the job table, and "Open run folder..." (also the 0.2 item "resume a run folder from the GUI").
- Acceptance: open a 14,000-molecule run folder in under 5 s and under 500 MB, scroll, search and select
  any molecule; a GUI batch of 14,000 runs with flat memory.
- 2026-11-05: `gui.molecule_list` (table model of light rows, search/status filter, LRU `ResultStore` of 12
  read back with `export.read_molecule`/`jobs.load_result` off the GUI thread), File > Open run folder...,
  Build > Resume run folder (stored settings, `jobs.stored_config`). Measured under xvfb on 2 shared cores
  with a synthetic 14,000-molecule folder (`scratch/make_synthetic_run.py`: real MMFF templates written with
  the engine's own writers): listed in 0.56 s, first molecule shown at 0.98 s, **+31 MB** RSS (+41 MB after
  browsing 60 molecules); search 40 ms, sort 60 ms, reading one result ~5 ms. The window itself is ~1.1 GB
  RSS before any molecule: torch (~500 MB, imported at startup by the GFN2 availability check,
  `engines.libcint_status` -> dxtb) and VTK under software OpenGL. Selecting a molecule takes ~0.4 s
  there, almost all VTK renders (~65 ms each, 3-4 per selection) under llvmpipe.
- Follow-ups: (1) a 400-molecule MMFF batch from the GUI grew RSS by 8 MB (~20-29 KB/molecule) with the
  list keeping rows only; ~5 KB/molecule of that is the light results `run_resumable` still returns,
  the rest is not yet attributed (the Log view up to its 20,000-line cap is a candidate). Find it, and
  let a GUI batch keep no per-molecule results at all (touches `dist/lanes.py`, after DIST-5). (2) Do not
  import torch at startup just to label GFN2. (3) One VTK render per selection instead of 3-4.

### IMPORT-2: Read Spartan '24 `.spartan` documents · in review 2026-11-05 (branch `feat/spartan-import`; acceptance pending)
- A `.spartan` file is an OLE compound document with one storage per molecule (`Molecules/M0001/...`):
  `Input.gz` holds the keywords, the label (CAS number in Brandon's files), charge and multiplicity and the
  Cartesian coordinates; `PropertyArchive.gz` holds about 50 scalar properties (`PROP VALUE NAME = value`:
  energies, dipole, HOMO/LUMO, CPK area/volume, ovality, QSAR ESP areas, polarizability) and per-atom blocks
  (Mulliken/NAO/ESP charges, atomic CPK areas). Checked on the PFAS QSAR files with `olefile` (pure Python).
- To do: `chem/io_spartan.py` + `quantabench-cli import *.spartan --table out.csv` (and the GUI import),
  property names mapped to QuantaBench's columns where they match; one row per molecule and file, keyed by
  label, so all 1,067 batch files become one table.
- Acceptance: one batch's values equal Spartan's own spreadsheet export; the 76-PFAS document imports whole.
- 2026-11-05: `chem/io_spartan.py`, `import *.spartan --table/--sdf/--conformers`, GUI routing. Checked on a
  100-molecule MMFF/DLFF03 conformer-search batch (labels match its input SDF), its extracted-conformers
  document and one small conformer document. Still open: (1) a Spartan spreadsheet export with values (the
  staged `.xlsx` was empty) to compare; (2) the 570 MB 76-PFAS document; (3) a document with the QSAR
  (quantum-chemical) properties, to confirm their Spartan names/units and fill `io_spartan.COLUMN_MAP`
  (HOMO/LUMO, dipole); (4) Spartan's own rule for the atoms its conformer libraries leave out (placed here
  by a rigid fit, typically ~0.5 Å from Spartan's placement).

### QSAR-1: First real test: the PFAS QSAR cohorts · after SCALE-1 and IMPORT-2
- Input: the 14,223 molecules of cohorts 1-6 (`data_engineering/Cohort */*.sdf`, CAS-labelled), one run per
  cohort or one run for all, split across NOVA and the MacBook with `grid build`.
- Protocol to agree with Brandon: conformers (RDKit or CREST) -> GFN2-xTB or AIMNet2 refinement in water ->
  QSAR descriptors for all; the QM stage (Psi4, his Spartan level) on a validated subset (VAL-1/VAL-2).
- Acceptance: one merged table for all 14,223 molecules, no manual recombination, and the overlap with the
  Spartan results compared.

### DIST-4: Per-stage routing · later
- Different stages of one molecule on different machines (CREST + Psi4 on the Mac, GPU refinement +
  LAMMPS on NOVA), handed over through run-folder artifacts.

### FORK-1: dxtb with tad-libcint 0.4 (GFN2 on Windows and macOS) · done 2026-11-05 (upstream PR pending Brandon's OK)
- tad-libcint 0.4.0 (2026-10-20) has Windows and macOS-arm64 wheels but needs tad-mctc ≥ 0.11; dxtb 0.4
  pins 0.7. First: open an issue/PR at dxtb. If it stalls: fork `brandonbtm29/dxtb` from v0.4.0 (branch
  `qb/tad-mctc-0.11`), adapt to the tad-mctc API changes, run dxtb's tests on all three OSes, and document
  `pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@<tag>"`. Supersedes WIN-1 when done.
- 2026-11-05: fork `brandonbtm29/dxtb`, branch `qb/tad-mctc-0.11`, tag `qb-v0.4.0.post1`; its CI is green on
  Ubuntu, Windows and macOS. Matches PyPI's stack to ~1e-8 relative. Install line in INSTALL.md;
  `engines.libcint_status()` names it when tad-libcint is missing or incompatible.

### FORK-2: tblite Windows wheels · done 2026-11-05 (upstream PR pending Brandon's OK)
- Upstream PR adding Windows wheels (cibuildwheel, MinGW-w64 gfortran); interim builds in
  `brandonbtm29/quantabench-wheels` attached to GitHub releases (`pip install tblite --find-links ...`).
- 2026-11-05: built in the fork itself instead (`brandonbtm29/tblite`, branch `qb/windows-wheels`, release
  `qb-v0.8.0-win1`, CPython 3.11-3.13 win_amd64, MSYS2 gfortran). NOVA reproduces the Linux energies to
  1e-14 Eh. The fork also runs tblite's own CI/docs/wheel workflows: disable those in its Actions settings.

### FORK-11: ROCm runtime hangs at process exit on Windows · ready (workaround in place)
- On NOVA (torch 2.9.1+rocm7.2.1), a Python process that used the GPU often never exits: `main()` returns
  and pytest prints its summary, then the process spins one core in the HIP runtime's teardown until it is
  killed (5 such processes on 2026-11-04/05). Workaround: `resources.hard_exit_if_needed` (`os._exit` after
  flushing) in `quantabench-cli`, the GUI, lane processes and pytest's `conftest.py`
  (`QUANTABENCH_HARD_EXIT=0|1` overrides). To do: a minimal repro (`import torch; torch.ones(1,
  device="cuda")`, then exit) and a report to ROCm/PyTorch; drop the workaround once fixed.

### FORK-10: ROCm SDK warning about paths with spaces · later
- On NOVA, AMD's Windows ROCm SDK prints "Unknown command line argument ...offload-arch.exe" when the venv
  path has a space ("GitHub Projects"): an unquoted path in its device-detection call. Harmless so far;
  reproduce outside QuantaBench and report to ROCm.

### FORK-5: CREST on native Windows · later
- Look into a win-64 build in the conda-forge crest feedstock (MinGW-w64). Low priority: WSL and the Mac
  (via DIST-2) already run CREST.

### FORK-7: pyvistaqt 0.13 crash upstream · ready
- Minimal repro of the `initializeGL` access violation with the AMD driver, filed upstream; re-check the
  `<0.13` pin on each pyvistaqt release (and test 0.13 on macOS).

### FORK-8: LAMMPS reliability on Windows and a Mac GPU build · later
- With MD-1: the same SMD input 20 times with the Windows installer's `lmp.exe` vs the PyPI wheel. Then
  `scripts/build_lammps_macos.sh` for an OpenCL GPU-package build in mixed precision, with a measured
  speedup over the Mac's CPU.

### FORK-9: OpenMM on the Mac's GPU · later (0.4 MD)
- Benchmark the beta `openmm-metal` plugin against OpenCL; go/no-go note in PLATFORMS.md.

## Needs Brandon (hardware / Windows / decisions)

- **HW-1 · done 2026-10-28:** ROCm 7.2.1 PyTorch 2.9.1 on BTM NOVA sees the RX 9060 XT (matmul
  6.1 TFLOP/s fp32). But: no QuantaBench engine runs on it yet (see GPU-1, GPU-2), and AMD's Windows
  wheel ships a slow CPU BLAS (52 vs 881 GFLOP/s for PyPI's CPU torch), so CPU work got ~17x slower.
- **HW-2 · done 2026-11-05:** dxtb and AIMNet2 on the RX 9060 XT (single + batched), energies CPU vs GPU:
  see GPU-1 (AIMNet2 ~10x faster on the GPU) and GPU-2 (dxtb works but is 5-8x slower there).
- **HW-3 · needs-Brandon:** GUI on Windows 11. Done 2026-10-27: the full suite including all GUI tests
  passes natively on BTM NOVA (Radeon RX 9060 XT, OpenGL 4.6) with pyvistaqt 0.12; pyvistaqt 0.13
  crashed (now pinned `<0.13`). Still to check by hand: HiDPI picking feel, 20k-atom slab frame rate.
- **HW-4 · needs-Brandon:** LAMMPS `in.smd` with the GPU (OpenCL) build on Windows.
- **DEC-1 · needs-Brandon:** approve `docs/SIMULATE_DESIGN.md` (after RES-1).
- **VAL-1 · needs-Brandon:** Spartan Essential 76 PFAS comparison needs his Spartan exports (SDF/CSV of
  ωB97X-D geometries and properties) placed somewhere the runs can read.

## Later (from ROADMAP)
Orbital isosurfaces, NMR (PySCF), properties inspector,
align/overlay, constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
substructure search over results, force-field assignment (OpenFF/GAFF2/OPLS), cross-molecule GPU
batching, process-pool CPU stage.
