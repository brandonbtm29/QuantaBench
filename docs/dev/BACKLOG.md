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

## VAL-1: Validation tooling against Spartan · in review 2026-11-05 (branch `feat/validate`; the 300-molecule run needs-Brandon)
Done (docs/VALIDATION.md):
- Spartan wide table: `inchikey`, `canonical_smiles` of the structure as Spartan drew it (formal charges from
  its bond orders), `structure_charge` + `charge_check`, `spartan_solvent` as QuantaBench's name
  (`spartan_solvent_raw` keeps `WASSER`), and one row per ambiguous structure (`label_resolved` empty,
  `label_source` = ambiguous, `cas_candidates`).
- `quantabench-cli sample`: reproducible validation set stratified by cohort x charge class (300, seed
  20261105), DFT-water first within a stratum, spread over size bins, AIMNet2's elements, <= 60 heavy atoms.
- `quantabench-cli validate`: joins by InChIKey / canonical SMILES / stereo-free InChIKey with a CAS
  cross-check; gas vs PM6, water vs DFT-water, solvation shifts flagged as mixed levels; N, MAE, RMSE, MSE, r,
  ρ, fit and outliers per pair, cohort and charge class; validation.csv, join.csv, summary.csv, report.md,
  parity plots.
Open (needs-Brandon):
- Re-import the 1,067 documents (`import --wide`) so every row has its keys, the ambiguous structures their
  rows and the charge check (the staged table predates them: 504 rows have no key from their `smiles`).
- Spartan's charge does not fit the drawn structure for some ions (C3A sample: 12 of 47, cations calculated as
  -1, phosphate dianions as -1); recalculate them or leave them out (the sampler does).
- The default sample has 237 molecules with PM6 and 64 with DFT-water but only 1 with both: DFT-water exists
  almost only for the charged cohorts, which have no PM6. Decide whether to add PM6 (or gas DFT) for those, or
  sample `--require dft` separately for the water comparison.
- Run the 300 in the gas phase and in ALPB water at GFN2-xTB (and AIMNet2), then `validate`. **Done
  2026-11-05:** grid NOVA + MacBook, gas 18 min, water 21 min, 298/300 ok in both phases. Findings and fixes are
  in VAL-3.

## VAL-3: Follow-ups from the 300-molecule validation run · in progress
- Done:
  - Ion dipoles about the centre of mass in the Spartan import (Spartan's are about its frame origin; up to
    10.5 D off).
  - tblite's electronic dipole stored for every conformer.
  - At Spartan's own geometries, GFN2/ALPB matches B3LYP/C-PCM water dipoles to 0.41 D for anions
    (r 0.997) and 1.9 D for cations (`scratch/dipole_fixed_geom.py` on NOVA). The remaining 4.2 D MAE comes
    from the conformer each program picks.
- Open:
  - `validate --boltzmann`: compare `dipole_debye_boltz` with a Boltzmann average over Spartan's DFT conformers
    (the `*_DFT_Top5` documents hold 5 per molecule).
  - The same fixed-geometry test for the gas phase against PM6 (r 0.62-0.67; C1B 0.18). It should show whether
    PM6 or the conformer is the cause.
  - GFN2's LUMO sits about 5.5 eV below PM6's and its HOMO 1.1 eV below, so its gap is 4.4 eV smaller
    (r 0.80-0.88). Document this as a systematic offset for QSAR: use the values as correlated descriptors,
    not as absolute values.
  - The Psi4 B3LYP subset on QuantaBench geometries (VAL-2) closes the loop on method vs geometry.
  - Rebuild the 300 (gas and water) with 0.3.0.dev3 and re-run `validate`: the 2026-11-05 tables have the old
    `sasa_A2` (zero atom radii) and no `cpk_*` columns. needs-Brandon (now, or with the full run).
- Done 2026-11-06 (areas and volumes, from Brandon's screenshots of Spartan's Preferences):
  - Spartan's CPK area and volume are the union of spheres with its VDW Radii table (carbon 1.92 Å vs Bondi's
    1.70). `cpk_area_A2` / `cpk_volume_A3` reproduce CPKAREA to 0.5 % and CPKVOLUME to 0.1 % (60 C3A B3LYP
    documents); Bondi radii give −6.8 % volume, RDKit's `volume_A3` −4.1 %.
  - `sasa_A2` used FreeSASA's ProtOr radii, 0 for non-protein atoms: fixed (Bondi + 1.4 Å probe).

## VAL-2: ωB97X-D/def2-SVP vs Spartan for the 76 PFAS · in progress 2026-11-06 (branch `feat/val2-essential76`; the runs needs-Brandon)
- The QM stage reproduces Spartan's level (Spartan used 6-31G*: run `wb97x-d/6-31g*` too). Compare
  dipoles, polarizabilities, 0.002 au volumes/areas and ESP extrema with Spartan's exports; recommend a
  default level for the 4,700-PFAS descriptor run with timings (PFBA anion: ~4.5 min on 2 threads with
  PCM, cubes and finite-field α).
- Done 2026-11-06 (docs/VALIDATION.md, "VAL-2"):
  - The Essential 76 documents are a multi-step conformer search (MMFF -> DLFF03 -> Spartan's neural-network
    ωB97X-D/6-31G* geometry DLFFG1 -> ωB97X-D/6-31G* single points -> DLXD2XV ranking): 63 of 76 completed
    (1,006 conformers); 13 failed (I, Si, P, a Na salt: outside Spartan's neural networks; PFOS drawn as the
    anion, calculated as neutral). Spartan's ωB97X-D/6-31G* values are gas-phase energies and dipoles per
    conformer. HOMO/LUMO, ESP extrema, density-surface area/volume and EST_POLARIZ at that level exist for one
    molecule (perfluorooctane, a C-PCM water optimization), in all 1,067 documents.
  - Import fixes: `MoleculeN` labels resolved by structure (76 wide rows, not 847); the Est. DFT rows'
    ωB97X-D/6-31G* SCF dipole (`QM_DIPOLE`) as `dipole_debye` + `qm_dipole_level`; their best row by
    `POSTENERGY`; `validate` pairs `qm_dipole_debye` with that level (`gas_dipole_qm_vs_dlxd2xv`).
  - Pilot at Spartan's geometry (TFA, PFPrA, triflic acid, PFHpA): Psi4 = Spartan to 5e-5 Eh (exact integrals)
    or 4e-4 Eh (density fitting) and 0.001 D. Timings scale as nbf^2.5. In PCM water, the finite-field α is three
    quarters of the cost.
  - QM stage fixes found on the way: scratch cleaned after a failed free-atom SCF; `run_driver` stops a driver
    that hangs after writing its result.
- Open (needs-Brandon): run the stages in `scratch/val2/run_val2_nova.ps1` (and `run_val2_mac.sh`) when NOVA is
  free, then pick the default level from the report.
- Open (code): an option to compute the QM polarizability in the gas phase (analytic) when the SCF is PCM, or to
  skip it. Under PCM it is about 70 % of the QM stage, which takes the 4,700 run from about 1-2 weeks on NOVA in
  water to about the gas-phase cost.
- Open: the full import's wide table (`spartan-qsar-v5`) has 771 `MoleculeN` rows from the Essential 76
  conformer documents. Re-import with this branch. Spartan reference values for HOMO/LUMO, ESP and areas at
  a DFT level exist only at B3LYP/6-311+G** C-PCM (the charged cohorts); a Psi4 B3LYP/6-311+G** PCM run at
  Spartan's geometries would compare those (needs ESP/density-surface pairs in `validate`: QSAR_ELPOT_MIN/MAX,
  QSAR_ELPOT_AREA/VOLUME, polar areas vs `qm_surf_*`).

## SETTINGS-1: A settings dialog as detailed as Spartan's Preferences · ready (Surfaces & radii, Display done 2026-11-06)
Review of Spartan '24's eleven tabs: Brandon's Claude Doc "Spartan Preferences review and QuantaBench settings plan".
Rule: a setting that changes a computed value is a science setting (`jobs.SCIENCE_FIELDS`, sent to grid
workers by `dist.worker.validate_config`); display settings never rename or rescale exported columns.
- Done: **Surfaces & radii** (Spartan's Settings > Surface and VDW Radii): radii set, 111-element table with
  per-element values, SASA probe (`settings.surfaces`).
- Done: **Display** (Document), 2026-11-06: energy and relative-energy units (kJ/mol by default, kcal/mol, eV,
  hartree), significant digits and number style (`settings.display`, `quantabench.units`). Applied to the conformer
  table, the molecule list, the info and thermochemistry text, the energy windows (stored in kcal/mol, exact until
  edited: `gui.energy_widgets.EnergySpinBox`) and the Surfaces panel. Exports keep `_kcal`, `_eV` columns. Left:
  reaction-energy units (nothing computes reaction energies yet); the CLI's printouts and the Properties page's
  raw descriptor list keep the column units; orbital energies stay in eV.
- Next, one tab per change, each with a GUI test like `tests/test_gui_settings_surfaces.py`:
  - **Compute** (Spartan's Jobs): an Expert box hiding the lane spec, GPU linear algebra and dxtb-on-GPU; a
    "N cores, M logical" line.
  - **Build defaults** (Settings > Import Filters, Conformer Rules): strip water, HCl and named counter-ions
    before the largest-fragment rule (science settings); a ring-conformer option for the RDKit search.
  - **Lookups** (Databases, URLs): URL templates with `%inchikey`, `%inchi`, `%cas`, `%smiles`, `%name`
    (PubChem, NIST WebBook by CAS, nmrshiftdb, EPA CompTox), opened from the molecule list; PubChem cache folder
    and maximum age; a folder of reference tables with experimental values.
  - **Machines** (Available Servers, Embedded Server): the `dist` settings (workers, port, bind, token), now
    only in settings.json.
- Later: **Molecule display** (default 3D style, atom labels such as charges and exposed area, surface style; the
  radii set for space-filling and the charge-ESP surfaces, which use Bondi now), **Toolbar** (after UI-1),
  **Miscellaneous** (CSV decimal separator and line endings for Excel, an HTTP proxy for lookups).

## Platform and multi-machine track (plan: docs/PLATFORMS.md)

### WIN-2: xtb and MOPAC on Windows disagree with the Linux reference values · ready
- First full run of the external-engine tests on NOVA (2026-11-05, conda envs `qb-tools`/`qb-psi4` from
  `install_windows.ps1 -WithEngines`): 4 failures, all values the program prints, not QuantaBench's math:
  - xtb 6.7.1 (conda-forge win-64 build of 2026-06-10): water GFN2 dipole z -0.332 e·Å instead of -0.474
    (`test_eng_xtb.py::test_real_singlepoints_match_xtb_printout`); water in water `gfn2_dipole_debye` 2.032
    instead of 2.546 (`test_real_properties_water_in_water`). Energies and forces passed.
  - MOPAC 23.2.5: PM7 water `pm7_polarizability_iso_A3` 1.347 instead of 1.259, methanol 3.22 instead of
    3.0 ± 0.2 (`test_eng_mopac.py`); heat of formation, dipole, HOMO/LUMO and COSMO area/volume passed.
- xtb: **fixed 2026-11-05.** xtb writes `"program call"` with raw Windows backslashes, so `xtbout.json`
  was invalid JSON. Dipoles and gaps fell back to the charge-only values. `eng_xtb.read_xtbout` now escapes
  them, and the xtb tests pass on NOVA.
- MOPAC: open. On NOVA, MOPAC 23.2.5 prints `ISOTROPIC AVERAGE ALPHA = 9.09001 A.U.` for the test's water,
  where the reference output has 8.49573: a version difference in TDHF POLAR, not a parsing problem. Record
  the version that produced the reference and make the test version-aware. Until then, don't mix MOPAC
  polarizabilities from different versions in one QSAR table.

### PLAT-1: First run on the MacBook · needs-Brandon
- INSTALL.md's macOS steps on the MacBook: `doctor --bench`, full suite, `pytest -m external` with
  `envs/tools.yml` + `envs/psi4.yml`, GUI by hand (HiDPI picking, trackpad pinch/rotate, dark mode).
  Record chip, cores and RAM in PLATFORMS.md section 2, and anything that breaks as new items.
- 2026-11-06: full suite and `-m external` done over SSH (PLATFORMS.md section 5: 841 passed; the real failures
  were fixed on `fix/macos-checks`, except LAMMPS, which needs `pip install mpich` on the MacBook, and MOPAC's
  polarizability, WIN-2). AIMNet2 on `mps` checked against the CPU. Left for Brandon: the GUI by hand.

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
- 2026-11-06: GitHub stopped running `macos-14` after 2026-11-05 19:15 UTC (jobs cancelled in 2 s,
  "configuration 'macos-14-arm64-us-default' is not supported"); both workflows now use `macos-15`. The GUI
  tests now run on macOS too (they had always been skipped there for want of `$DISPLAY`). MOPAC: the
  MacBook (OpenBLAS) and NOVA (MKL) both print 1.347, the cloud's Linux build (MKL) 1.259, all 23.2.5 build 0,
  so BLAS is not the cause: the Linux build differs from the macOS and Windows builds.

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
  list keeping rows only; ~5 KB/molecule of that is the light results `run_resumable` still returns.
  Let a GUI batch keep no per-molecule results at all (touches `dist/lanes.py`, after DIST-5). (2) Do not
  import torch at startup just to label GFN2. (3) One VTK render per selection instead of 3-4.
- 2026-11-05 (branch `perf/gui-startup`), same synthetic folder and machine:
  - (2) done: no torch/dxtb/tad-libcint/aimnet/ovito in the GUI process at start-up. Labels use
    `engines.available(key, quick=True)` (versions and files, `libcint_status_from_versions`); GPU list and
    the full libcint check run in a child process (`resources.detect_isolated`) whose verdict relabels GFN2.
    Idle RSS with the 14,000-molecule folder open: **1,133 -> 634 MB** (window alone 1,102 -> 602 MB). VTK
    is ~355 MB of that (libraries ~155 MB, llvmpipe context and buffers ~200 MB); without VTK ~280 MB. So
    the 500 MB target holds only without VTK under software OpenGL; on a GPU the buffers leave RSS. Check
    the number on NOVA and the Mac before deciding on more.
  - (3) done: one render per selection (was 3-9: `Plotter.reset_camera` renders even with `render=False`,
    the selection clear rendered, and pyvistaqt's `auto_update` re-rendered 5x/s while idle). Selection
    median **557 -> 224 ms**; scrolling 14,000 rows in 200 steps 10.0 -> 2.5 s. What is left of a selection
    is one render with SSAA: ~150 ms under llvmpipe, ~40 ms with FXAA (RSS about the same). Possible next
    step: FXAA when the OpenGL renderer is software (llvmpipe, WSL without GPU).
  - (1) attributed: a Log line costs ~7 KB RSS (laid-out text block, shown or not) and a GUI MMFF batch logs
    ~3 lines per molecule (~21 KB/molecule), so the log grew until its 20,000-line cap (~140 MB). Cap now
    5,000 lines (~35 MB). Still open: the light results above, and an MMFF batch imports torch at its start
    (`jobs.run_resumable` and `pipeline.run` call `resources.select_device` before knowing whether the
    method needs torch: +~500 MB for the rest of the session). Select the device lazily, or only for
    torch engines (chem/jobs.py, chem/pipeline.py).
- 2026-11-05 (branch `fix/runtime-guards`), for the 14,223-molecule split across NOVA and the MacBook:
  - `lanes auto` plans a GPU lane only where this machine's benchmark (`<method>@<gpu>` vs `<method>@cpu`)
    says the GPU is the faster lane; MacBook AIMNet2 (6,387/h CPU, 2,978/h MPS): CPU lanes only (6 x 2 with
    the default 13 threads). Without a benchmark the GPU lane as before, and the reason says so. Still to check: a lanes benchmark on the Mac
    (`machines --bench aimnet2` with `--lanes`), since 1 MPS lane + 4 CPU lanes might still beat 6 CPU lanes
    if MPS and the CPU do not contend (the rule compares one lane with one lane).
  - Benchmarks in water (`machines --solvent water`: `<method>+water@<device>`) and `--bench-set mixed`;
    readers of the keys (speed-aware lanes, `lanes auto`, grid and chunk weights) use the run's solvent.
  - Settings > Compute: memory per lane and time limit per molecule.

### IMPORT-2: Read Spartan '24 `.spartan` documents · in review 2026-11-05 (branches `feat/spartan-import`, `feat/spartan-qsar`; acceptance pending)
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
- 2026-11-05 (`feat/spartan-qsar`): the 129,164 molecules a full run logged as "no Input.gz" (All_Conformers,
  SemiEmp_PM6 and base documents) come from Spartan for macOS, which writes the same streams uncompressed
  (`Input`, `PropertyArchive`); both are read now and anything still unreadable is counted by reason. Rows
  carry the level of theory (`Commands`/`Calculations`) and the job status (`JobLog`); PM6/DFT HOMO/LUMO
  (hartree -> eV), dipole, ZPE, H/G corrections and entropy fill `COLUMN_MAP` (units checked against the
  output, the manual and the archive's own identities). Labels Spartan made up (`M0003`, `M0018[1]`) are
  joined to the base document by structure (`label_source` input/structure/ambiguous/unresolved); `--wide`
  writes one row per label with each level's best result under a level prefix. Staged sets: 2,750 of 2,750
  molecules read; derived rows 2,480 resolved, 120 ambiguous (6 anions of C3A_P4 under two CAS numbers each),
  0 unresolved. Still open: (5) the optimized geometry of Spartan OPT jobs (Input keeps the starting one;
  probably in `Archive.carc`); (6) ambiguous structures (an anion under the CAS numbers of its salts): share
  the result with all its labels in the wide table, or leave it out (now); (7) a full-run timing on NOVA.

### QSAR-1: First real test: the PFAS QSAR cohorts · after SCALE-1 and IMPORT-2
- Input: the 14,223 molecules of cohorts 1-6 (`data_engineering/Cohort */*.sdf`, CAS-labelled), one run per
  cohort or one run for all, split across NOVA and the MacBook with `grid build`.
- Protocol to agree with Brandon: conformers (RDKit or CREST) -> GFN2-xTB or AIMNet2 refinement in water ->
  QSAR descriptors for all; the QM stage (Psi4, his Spartan level) on a validated subset (VAL-1/VAL-2).
- Acceptance: one merged table for all 14,223 molecules, no manual recombination, and the overlap with the
  Spartan results compared.

### QSAR-2: Follow-ups from the 325-molecule stress sample (2026-11-05) · ready (decisions: needs-Brandon)
Fixed on `fix/stress-sample`: metal complexes refused instead of computed without their metal, Hg/Sn bonds
kept, counter-ions/charge/multiplicity/force field in the notes and `summary.csv`, SDF record errors,
`--id-col` for SD files, placeholder atoms, unphysical xTB polarizabilities, the Warp CUDA-driver line.
Still open:
- **Protonation rules on polyamines and phosphonates** (chemistry): every aliphatic amine is protonated
  independently, so piperazines and 1,4-dimethylpiperazine come out +2 at pH 7 (pKa2 about 5.3 and 3.8:
  should be +1); every P-OH is deprotonated with the 1st-pKa rule (alkylphosphonic acids pKa2 about 7.5).
  Needs a site-interaction rule (no second protonation within 2-3 bonds of a charged site) or dimorphite-dl
  as default; 77 of 325 records change charge at pH 7, most of them correctly.
- **Al3+/Zn2+ salts** (needs-Brandon): monoatomic Al3+ or Zn2+ next to organic anions (fosetyl-Al) now
  fails as a metal complex. If Brandon treats them as counter-ions like Na+, add them to
  `standardize.COUNTERION_METALS` (one line; the anion is then computed).
- **Metal complexes that need a 3D builder**: Cr(III) azo dyes have no UFF parameters (method `mmff`
  fails; xTB methods refine unoptimized ETKDG geometries). Cobalamins, carbonyls and Rh/PPh3 complexes
  drawn as ions fail by design. A route: GFN-FF pre-optimization (xtb `--gfnff`) instead of UFF, or a
  coordination-geometry builder.
- **Per-molecule time limit**: done on `fix/memory-time-guards` (2026-11-05). `resources.molecule_timeout_s` /
  `build --molecule-timeout`, default 1800 s. Lanes are killed and restarted past it. One process stops the
  molecule at its next checkpoint (best effort). The molecule is failed `time limit (N s) in <stage>`, and
  `--retry-failed` redoes it. Still open: RDKit embedding has no checkpoint (ETKDG `EmbedParameters.timeout`
  since RDKit 2024.03 would bound it in one process too).
  Done on `fix/runtime-guards` (2026-11-05): the embedding gets `conformers.EMBED_TIME_FRACTION` (1/4) of the
  limit, in one process and in lanes. RDKit's timeout ends a call with a single conformer however many it had
  finished, so one conformer is timed first and fewer are asked for when they would not fit; the timeout is
  the hard stop. Notes say when it ran out and with how many conformers. Still open: the MMFF optimization of
  hundreds of conformers of such a molecule has no checkpoint either; the pipeline could pass
  `timeout_s=confs.embed_time_limit(cfg.molecule_timeout_s)` explicitly (today: `resources.set_molecule_limits`).
- **Ids of the baseline cohorts** (needs-Brandon): cohorts 1-3 are titled by CAS number, cohorts 4-6 by a
  name that is sometimes cut off after a word, with the CAS in `CAS_RN`. `--id-col CAS_RN`
  now gives CAS ids for those; decide which the merged QSAR table uses.
- **Transition-metal spin states**: an odd-electron complex without drawn radicals is computed as a
  doublet (Cr(III) is d3, usually a quartet), with a note. Tables need optional `charge`/`multiplicity`
  columns (MoleculeInput has the fields; the CSV/Excel readers do not read them) so such records can be
  given their spin state. dxtb on that complex from unoptimized ETKDG geometries (no UFF parameters)
  ran for many minutes.
- **Broken records with radicals**: an amine-borane record is stored with a bare B (2 unpaired electrons,
  +1); it is now computed as a triplet cation with a "check the record" note. A heuristic for
  hydrogen-deficient B/N/C could fail such records instead.
- **Memory of dxtb on large molecules** (highest priority for 14,000 molecules with lanes): four processes
  were killed by the memory cgroup at 2-3.9 GB resident: the MMFF run at a 90-heavy-atom cyclic peptide and
  a +2 alkaloid in the GFN1-xTB properties step (dxtb charges/polarizability, autograd field response),
  the gfn1-xtb run on a Cr(III) azo complex, and the tblite run in that properties step. The job table recovered (crash counted, molecule
  moved last), but on NOVA with many lanes this can kill lanes repeatedly. Measure memory vs atom count and
  skip (with a note) or chunk the polarizability above a size; or use tblite's finite-field polarizability.
  Done on `fix/memory-time-guards` (2026-11-05): `chem/memory_guard.py`. The cause is dxtb's `scf_mode="full"`
  (memory goes as SCF iterations x nao²). An estimate per step is checked against a per-lane budget
  (`--memory-budget`, default available / lanes, at least 1.5 GB). Over budget, the polarizability comes from
  tblite (the same value), or is skipped with a note; a refinement that does not fit fails with the reason, or
  `auto` moves on to its next engine. Measurements are in docs/ENGINES.md. Still open: the rule under-estimates
  hard SCF cases it cannot see (closed shell with near-degenerate levels: 5.8x the neutral run for a peptide
  with two electrons removed); calibrate on NOVA's ROCm torch and on the MacBook, whose allocators may differ.
  Done on `fix/runtime-guards` (2026-11-05): the frequencies stage (`memory_guard.frequencies_plan`, one call at
  the start of `vibrations.analyze`'s Hessian step). The analytic Hessian grows as atoms x nao² (TFA anion
  353 MB, PFBA anion 1.2 GB, PFOA anion > 6 GB); over the budget it becomes finite differences (one energy +
  forces of memory, about the same time), and finite differences or IR dipole derivatives that do not fit are
  skipped with a note. Measured only up to 2 GB resident here (other agents on an 8 GB VM); the GFN2 and the
  larger points of the rule are extrapolated: check a few on NOVA and the Mac. The pipeline could pass
  `memory_budget_gb=memory_budget(cfg)` to `vibrations.analyze` explicitly (today: the lane's share through
  `resources.set_molecule_limits`, auto in one process).
- **The xTB properties step on dications** (a piperazine, +2 at pH 7) took minutes of GFN1-xTB SCF in the
  MMFF run; measure and cap its SCF iterations.
- Settings isolation: during this work `~/.config/QuantaBench/settings.json` (and `QuantaBench.conf`) were
  rewritten with test-like values (pH 0, 1 conformer, 1 K) by another process on the machine; check
  that no test or GUI path writes there when `QUANTABENCH_SETTINGS_DIR` is set.

### QSAR-3: Surface tension model on QuantaBench descriptors · ready (data pull needs-Brandon)
Plan: docs/SURFACE_TENSION.md. Two targets, never pooled: pure-liquid σ at 25 °C for compounds that are liquid
there, and γ_CMC/CMC of PFAS surfactants in water.
- Data: `build --phases gas,water` on the QSAR cohorts, then on NOVA (PubChem is blocked in the cloud
  environment) `pubchem enrich RUN --validate --experimental --records experimental.csv`. Add Jasper (1972)
  coefficients (via the MIT `chemicals` package) and NIST ThermoML (free bulk download) for pure liquids, and
  curated literature γ(c) data for PFAS (Brusseau 2019 Table S1 sources), each with the same provenance columns
  as the PubChem records.
- Baselines: group-contribution parachor and a parachor fitted on QuantaBench volumes, σ = (Pρ/M)⁴; TEST
  predictions from CompTox.
- QSPR: α, α/V, vdW volume, SASA, μ, μ²/V, ESP (GIPF) descriptors, polar surface, F/C and fluorinated surface
  fraction, dG_solv. Ridge/PLS first, then GPR/boosting.
- Validation: leave-cohort-out, leave-class-out, leave-chain-length-out; applicability domain by leverage,
  kNN distance and F/C range; flag compounds solid at 25 °C.
- Acceptance: a training table with provenance per value, and a model report (MAE/RMSE vs the baselines per
  split).
- Needs Brandon: the target choice (pure-liquid σ vs γ_CMC first), 20 or 25 °C, and access to DIPPR/TRC for
  validation.

### PUBCHEM-1: PubChem follow-ups · ready
- **`build --pubchem-validate`** (not done: it would touch the pipeline's summary row, which other branches are
  changing). Add `pubchem_cid`/`pubchem_match` to each molecule's summary row when online, never failing a
  molecule when PubChem is unreachable. Today: `pubchem enrich RUN --validate` after the build gives the same
  columns.
- **Annotation index:** `pug_view/annotations/heading/Surface%20Tension/JSON?page=N` lists every compound with a
  surface-tension annotation in a few requests (paged). Use it to fetch PUG View only for CIDs that have one,
  and to find surface-tension data for compounds outside the input list. Capture one page as a fixture first:
  the format is not covered by the current fixtures.
- **Refetch policy:** 404s are cached forever; add `--refresh-missing DAYS` for records PubChem adds later.
- **Parent CIDs:** `/compound/cid/X/cids/JSON?cids_type=parent` would tell a salt's CAS RN from its parent's
  without RDKit's largest-fragment rule.
- **Condition classes:** check the classifier on the full pull. List every `needs_review` surface-tension
  record, and extend the reference rules (pure-liquid tables vs surfactant literature) from what is there.
- **Trial findings (validation300, first 50, 2026-11-05):** 40 match, 1 stereo, 6 CAS not in PubChem (found
  by InChIKey), 3 different. Of the 3:
  - 93894-66-7: drawn with one F per chain carbon. The name says heptadecafluoro, so it is a structure error
    in the source.
  - 165951-17-7: PubChem's record is the dipropylammonium salt of HFPO-DA; the table has only the amine.
  - 218462-19-2: a mixture with dichloroethylene.

  The last two are now `component_of_mixture`. Feed `different_structure` and smaller-component cases back
  to the cohort curation before the QSAR fit.
- **CompTox (CTX API):** the same enrichment by DTXSID (batch POSTs of 1,000; free key): TEST's predicted
  surface tension and the experimental physchem values. Check the property names on the workstation first
  (api-ccte.epa.gov is blocked in the cloud environment).

### DIST-6: Grid weights from measured throughput · done 2026-11-06 (work sharing; learned weights optional)
- 2026-11-06: **work sharing** instead of learned weights: while `grid build` follows a grid, a machine that
  has finished takes molecules another has not started (`client.WorkSharing`, `POST /v1/jobs/<id>/release`,
  `jobs.Handoff`; docs/WORKERS.md), and every share runs its largest molecules first. In the 2026-11-05 water
  run the MacBook was idle for 8 of 21 minutes; with sharing that tail is about one molecule per lane. Left
  (optional): record molecules/hour per protocol from finished grids and use it for `--weights auto`, so
  fewer molecules move; measure the first real grid with sharing on NOVA + MacBook.
- Validation run (tblite GFN2, 50 conformers, top 5, frequencies), water phase: the MacBook finished its 169
  molecules in 12.5 min (811/h); NOVA took 20.8 min for its 131 (378/h). `--weights auto` had split them
  2680:2240 from the single-molecule benchmark. For the full run pass measured weights (about 2:1), and
  have `grid build` learn: record each machine's molecules/hour per protocol from finished grids, and
  prefer that over the benchmark in `--weights auto`. Work stealing (a machine that finishes early takes
  the other's pending molecules) would make this unnecessary.
- Lane layouts: both machines ran 6 lanes x 2 threads. Timed on 49 molecules (`scratch/lanes_test` on
  each machine, 2026-11-05):
  - Mac 6x2 181 s, 14x1 141 s.
  - NOVA 6x2 427 s, 12x1 363 s, 24x1 331 s, 12x2 318 s.

  `lanes auto` now gives tblite one lane per physical core with the threads shared out (c0427d9). On the same
  49 molecules the Mac is 2.3x NOVA, so pass `--weights brandons-macbook-pro=2.3,local=1` until grids
  learn their weights.

### UPD-1: Update button gaps · in review 2026-11-06 (branch `feat/update-packages`; worker updates: design only)
Brandon asked whether Help > Check for updates pulls "the latest packages and GitHub version". It pulled
the code, but three things never arrived through it.

1. **Installed packages were never upgraded.** pip reinstalled only when `pyproject.toml` changed, and kept
   every package that still fit. Done: `quantabench-cli update --upgrade-packages [--upgrade-strategy
   eager|only-if-needed]` and, in the app, "Also upgrade the installed packages" or "Upgrade packages...".
   - It works with or without new commits, keeps the installed PyTorch with a constraints file and keeps the
     fork packages on pip's command line.
   - It lists `pip list` before/after, and the app asks before restarting.
2. **Windows fork pins lived only in `install_windows.ps1`**, so a new tblite build (win2, PERF-2) could
   never arrive through Update. Done: `scripts/fork-packages.json`.
   - Per platform: `packages`, `only_binary`, `find_links` as {package: URL}. All three installers read it,
     with the same pip lines and stamps as before (checked by dry runs, the stamp files, and NOVA's
     PowerShell 5.1 parser).
   - `update` reinstalls when this platform's entry changes in a pull. It then force-reinstalls
     (`--no-deps`) a git-tag or release-page package whose requirement or link changed.
   - `--upgrade-packages` force-reinstalls those packages too.
   - Tests: tests/test_update.py, tests/test_install_scripts.py.
3. **Workers keep the old code.** A running `quantabench-cli worker` imported the code it started with. An
   update on that machine changes the files under it: new modules can load next to old ones in the middle
   of a job. A grid then mixes commits; `grid build` already warns about that. Not changed yet:
   docs/WORKERS.md says to restart workers after an update. Options, roughly in order of preference:
   - **a. `quantabench-cli worker --update-when-idle` / `POST /v1/update`** (token, Tailscale bind as
     today). The worker pauses (like `POST /v1/pause`) and lets the running job finish. It runs
     `update.update()` (fast-forward only; `upgrade_packages` only if the request asks), then restarts
     itself. POSIX re-execs with the same argv. Windows starts a new detached process and exits, because the
     Task Scheduler restart is unverified (WORKERS.md). The queue resumes, since jobs are resumable already.
     It refuses while a job runs unless asked to wait, and reports the new commit in `/v1/info`.
   - **b. `quantabench-cli grid update [--machines ...] [--upgrade-packages]`** calls (a) on each worker
     and waits until each reports `origin/main`'s commit. `grid build` could offer it when commits differ,
     instead of only warning, and the Machines dock could show "N workers run another commit".
   - **c. Minimal, no API:** `quantabench-cli worker restart` for the local machine. It sends SIGINT
     (worker already stops after the current molecule), waits for the port to free, and starts the worker
     again with the arguments saved in its state folder. launchd's KeepAlive already restarts a stopped
     worker on the Mac.
   - Either way, `update.update()` should first check for a running worker from the same checkout and
     venv (via `psutil`, or a pid file in the worker's state folder) and say so. Upgrading packages under a
     running worker on Windows also hits the locked-file problem below.
- Left:
  - **Windows in-app upgrades** run pip while the GUI has PySide6, numpy, VTK and the rest loaded. pip
    usually manages by moving the files aside, leaving `~...` folders in site-packages, but it can fail on a
    locked DLL. The message then says to close the app and use the CLI. A sturdier route is to finish the
    upgrade in a helper process after the GUI exits: a console window shows pip, then the GUI starts again.
    Try `--upgrade-packages` from the app on NOVA first (needs-Brandon), and clean up `~...` leftovers on
    the next start if they appear.
  - `--check --upgrade-packages` could preview the upgrade with `pip install --dry-run --report -`.
  - The worker update (3).

### PERF-2: tblite Windows wheel with OpenBLAS · ready
- One GFN2/ALPB energy+gradient (`scratch/tblite_scf_bench.py`, 1 thread): MacBook 13.5 / 32.5 / 81 ms
  for 31 / 62 / 110 atoms, NOVA 24.8 / 66.8 / 212 ms. The gap grows with size (1.8x to 2.6x). The macOS
  wheel links OpenBLAS; the fork's Windows wheel bundles reference BLAS/LAPACK.
- To rebuild with MSYS2's `mingw-w64-ucrt-x86_64-openblas` (`-Dtblite:lapack=openblas`), check first:
  - That package depends on LLVM's OpenMP (`-omp`), while tblite links libgomp, so two OpenMP runtimes
    would sit in one process.
  - It ships `openblas.pc` but no `lapack.pc`, which dftd4 and multicharge may look for.
  - Energies must match win1 to 1e-10 Eh.
- Expected gain: NOVA 1.5-2x on mid-size and large molecules, about 15 % off a two-machine full run.
- Rolling it out (UPD-1): point `windows.find_links.tblite` in `scripts/fork-packages.json` at the
  `qb-v0.8.0-win2` release, and change `chem/eng_tblite.py` `WINDOWS_WHEELS` to match
  (`tests/test_install_scripts.py` checks both). Help > Check for updates then reinstalls tblite on NOVA,
  even under the same version number. Give the wheel a new version (e.g. `0.8.0.post1`, pinned as
  `tblite==0.8.0.post1`) so rerunning the installer picks it up too. Its pip run does not replace a
  version that still fits.

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
  1e-14 Eh. tblite's own CI, docs and wheel workflows are disabled in the fork (2026-11-05); only `qb-windows-wheels` runs.

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
- **VAL-1 · in review 2026-11-05:** tooling done on `feat/validate` (see VAL-1 above); the Spartan wide table of
  the PFAS QSAR run is the reference. Still needs-Brandon: a re-import for the new key columns, the charge
  mismatches, and the 300-molecule gas/water run. The Essential 76 ωB97X-D comparison is VAL-2.

## Later (from ROADMAP)
Orbital isosurfaces, NMR (PySCF), properties inspector,
align/overlay, constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
substructure search over results, force-field assignment (OpenFF/GAFF2/OPLS), cross-molecule GPU
batching, process-pool CPU stage.
