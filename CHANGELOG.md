# Changelog

## 0.3.0.dev (unreleased)

### `conformers compare` reads every machine's share of a grid run (2026-11-13)
- **The bug.** Given a grid run's phase folder (or its grid folder), `conformers compare` read only this machine's run folder. The other machines' molecules were silently left out.
  - Those shares live in the grid folder's `parts/<machine>/<run folder>` when the grid build downloaded them, or in the output folder when SyncThing brought them.
  - In the 2026-11-13 conformer checks, 4 of 10 pilot molecules had run on the MacBook.
- **The fix.** `read_sources` follows `grid.json`'s shares to wherever this machine has them: the downloaded archive first, else the synced copy beside the grid folder or up to two levels above it. Each run folder is read once.
  - A share found nowhere is logged with its molecule count.
  - A share name from `grid.json` (a synced file) must be a plain folder name.
  - The log now gives each run folder's molecule count.

### Scratch never piles up; Psi4 memory per run, SCRATCH-1 (2026-11-13)
- **What happened.** In b03's ESP pass, lanes killed at the 30-minute per-molecule limit could not delete their Psi4 job folders. By morning NOVA's `%TEMP%\QuantaBench` held 59 of them with 479 GB of DF integrals (`psi.<pid>.97`, 13-21 GB each).
- **A failed job's folder** keeps its inputs and logs for inspection. Psi4's scratch files and anything over 8 MB are deleted, and the error says how much was freed.
- **Owner records.** Every scratch folder now holds an owner record (`.qb_owner.json`: pid, start time, host). `external.sweep_scratch` cleans the folders of processes that have ended:
  - their bulky files go at once;
  - the folder itself, with its logs, goes after 3 days;
  - a running job's folder, or another host's, is never touched;
  - a folder without an owner record (older versions) counts after 12 hours without a change.
- **When the sweep runs:**
  - the lane supervisor sweeps a lane it killed (time limit) or found dead, at once, and every lane's when the pool closes;
  - every run sweeps when it starts.
  - A log line says what was freed: `Scratch of lane cpu3 (302-27-2): 15.1 GB freed in 1 left-behind scratch folder(s) ...`.
- **`quantabench-cli scratch`** shows the scratch folder (running, left behind, without an owner record) and `--clean [--dry-run] [--older-than DAYS]` cleans it. Settings > Engines > Keep job folders also turns the sweep off. The Psi4 driver calls `psi4.core.clean()` after a failure as well.
- **`--qm-memory GB`** (`BuildConfig.qm_memory_gb`, a resource, not a science field) sets each Psi4 job's memory for a run, over Settings > Engines > QM memory.
  - The auto lane plan fits the QM lanes to the RAM. Lanes capped by memory now share the thread budget: NOVA with 4 GB per job ran 5 lanes × 2 threads of 19, and now runs 5 × 3.
  - Too little memory makes DF-SCF keep its integrals on disk (DiskDFJK), many times slower: NOVA timed out on 55 of 65 large molecules.
  - The driver now says so before the SCF, with the size: naux · nbf(nbf+1)/2 · 8 bytes, 12.1 GB for 784/4,901 basis functions.
  - `qm["df"]` records `nbf`, `naux`, `tensor_gb` and `algorithm`, and a warning notes a disk-based run.
- **Tests:** `QUANTABENCH_SCRATCH_DIR` points the scratch root elsewhere when the settings name none. The test suite sets it, so a test's run never sweeps the real scratch folder.

### QM stage: switches for an ESP study, and imaginary modes at the QM level (2026-11-12)
- **`--no-qm-polarizability`** (BuildConfig `qm_polarizability`, a science field; the QM page's new Polarizability
  box): skips the static polarizability of the best QM conformer. With DFT the CPSCF (or finite-field) SCFs are the
  slowest part of the properties step: most of b02's 31-minute ESP pass. Dipole, charges and orbitals stay.
  **`--qm-properties/--no-qm-properties`** switches the whole properties step from the command line (it was only
  in the GUI). A run folder from before keeps its polarizability (the default).
- **`--no-keep-cubes`** (`qm_keep_cubes`, files only; the QM page's "Keep the cube files"): deletes the density
  and ESP cubes (~16 MB per molecule) once the surface descriptors and the 5 kJ/mol ESP profile are saved. When
  the surface analysis did not finish, the cubes stay, with a note. The default keeps them, as before: the
  Surfaces panel draws the QM surface from them. For an ESP pass over many molecules:
  `--qm wb97x-d/6-31g* --cubes --no-qm-polarizability --no-keep-cubes`.
- **Imaginary modes in the QM frequencies** (`--qm-freq`): a mode below -20 cm-1 left after the tight
  re-optimization is removed as in the pipeline's own frequency step (`freq_imag_retries`, default 3): Psi4's
  driver steps 0.15 Å (largest atom) along the lowest mode both ways, re-optimizes both, keeps the lower and
  computes a new Hessian. `qm["vibrations"]` records `imag_found_cm1`, `imag_steps`, `imag_left_cm1`,
  `imag_energy_change_ev` and `imag_max_shift_A`, and a note says what was done.

### The Analyze table: molecules x properties, a plot, formula columns and fits, UI-3 (2026-11-12)
- **Table dock** (Analyze, in front at the bottom, with Results and Spectra as tabs): one row per molecule of the
  Molecules dock, the same rows in the same order, virtualized (14,000 rows build, sort and filter in well under a
  second). Columns: #, ID, name, status, formula, conformers, best energy (in the unit of Settings > Display),
  method, SMILES; the data fields an imported file posted; every summary and descriptor value of the built
  molecules; and formula columns. The values come from the results in memory (a batch's finished molecules,
  imports, results read from their folders) and, for File > Open run folder, from the folder's summary.csv and
  descriptors.csv, read on a worker thread (the job table fills in molecules those files miss, as in an
  interrupted batch).
- **Columns...** shows or hides columns (remembered), grouped by where they come from, with the name each one has in
  formulas. Sort by any column (numbers as numbers, missing values last), filter by ID, name, SMILES, formula or
  method. Selecting a row selects the molecule in the Molecules dock and the other way round (a molecule hidden by
  the Molecules dock's search clears that search).
- **Copy and export:** Ctrl+C copies the selected cells as tab-separated text for Excel, Ctrl+Shift+C with the
  column titles; Export writes the rows and columns shown to CSV (UTF-8 with a byte-order mark, so Excel reads µ
  and Å) or to an Excel workbook when openpyxl is installed (otherwise the menu entry says
  `pip install openpyxl`). Numbers keep 15 significant digits.
- **Formula columns** (New column..., or right-click a header to edit or remove one): `+ - * / **`, parentheses,
  numbers, column names and `log`, `log10`, `exp`, `sqrt`, `abs`, `min`, `max`, `pow`, evaluated row by row with
  NumPy; a missing value gives an empty cell. Nothing else is accepted (no attribute access, indexing, other calls,
  comparisons, lambdas, comprehensions, f-strings or `__` names), and the text never reaches `eval` or `exec`
  (`analysis.formula`: an `ast` whitelist and a tree-walking evaluator). A column whose name is not an identifier
  is referenced by its alias (`E (kJ/mol)` -> `E_kJ_mol`, `σ` -> `sigma`), shown in the column chooser; the best
  energy is `best_energy_eV`, in eV.
- **Fit...** fits the parameters of a formula (its names that are not columns, e.g. a, b, c in the quantum parachor
  `((a*alpha + b*mu**2 + c)/Vm)**4`) to a target column with `scipy.optimize.least_squares` on the rows the table
  shows where every value is present: the parameters with standard errors, n, MAE, RMSE and R², and Add as column
  puts the fitted formula in the table.
- **Plot dock** (Analyze, in front on the right, with Surfaces and Appearance as tabs): any two numeric columns,
  colour by a third (viridis), a least-squares line with slope, intercept, R² and n; the table's filter filters it;
  clicking a point selects that molecule, which is ringed.
- **Kept:** the column choices, formula columns and plot settings are kept in the window settings until the project
  document (UI-2) holds them. The workspace layouts' version is now 2: layouts saved by the UI-1 build are dropped
  once, and Analyze starts with its new default.

### The app shell: task workspaces, Appearance, playback bar, status line, Library, UI-1 (2026-11-12)
- **Workspaces:** an activity bar on the left switches the window between **Build**, **Calculate**,
  **Simulate**, **Analyze** and **Library** (Alt+1 ... Alt+5, Control+1 ... Control+5 on macOS, or View >
  Workspace; Ctrl+1 ... Ctrl+3 stay the mouse modes). A workspace is an arrangement of the same docks around the
  shared 2D/3D view: Build shows the molecules and Appearance; Calculate the pipeline, Machines and Appearance;
  Simulate the surface builder and LAMMPS export (a stub page for the MD workspace to come); Analyze surfaces and
  spectra next to the conformer table; Library the run folders. Nothing is rebuilt on a switch: the selection, the
  view and every panel stay as they are.
- **Each workspace keeps its own layout:** a dock moved, closed or tabbed in Analyze stays that way in Analyze only,
  also after a restart (kept per machine in the window settings, like the window size). View > Reset layout
  restores the current workspace's default, View > Reset all workspace layouts every one. The dock layout saved by
  earlier versions is not carried over: the first start shows the defaults (Calculate, the closest to the old
  window).
- **Appearance dock:** style, labels, colours, hydrogens in 3D and 2D and the 2D view in one place, instead of the
  combo boxes on the view toolbar. Ctrl+D (show 2D), View > Hydrogens in 2D and the mouse modes stay where they were.
- **Playback bar** under the view: first / previous / play / next / last, a slider and "Conformer 3 of 6, ΔE 3.15
  kJ/mol, weight 15.2 %" for the selected molecule's conformers (1, 2, 4 or 8 per second; play wraps around). It
  and the conformer table follow each other.
- **Status line:** the number of molecules, the selected molecule and conformer, next to CPU threads and the GPU;
  on a GPU also the memory PyTorch has allocated (refreshed once a second, only once torch is loaded).
- **Library:** the run folders under the output folder (three levels deep), newest first, with their molecules
  (built, failed), method, phase and state (finished, *n* to do, running); filter and sort them, double-click or
  Open to list a folder's molecules, Resume to continue its unfinished ones. Read-only, off the GUI thread, scanned
  when the Library is first shown and on Refresh.

### Imaginary frequencies removed in the frequency step (2026-11-12)
- In batch b03, 75 of 1,000 molecules per phase ended with an imaginary mode below -20 cm-1. Two causes: a residual
  force on a soft torsion (90 %: -20 to -50 cm-1; the pre-optimization stops at fmax 0.01 eV/Å and most stopped at
  0.009), and saddle points kept by symmetry, such as a planar NH2 on an aromatic ring (-100 to -430 cm-1), which
  no tighter optimization leaves.
- `vibrations.analyze(imag_retries=N)` and the pipeline's `freq_imag_retries` (default 3; `--freq-imag-retries`
  for `build`, `grid build` and `freq`): when a mode is below -20 cm-1 the geometry is re-optimized to fmax 0.002
  eV/Å; while one is left, the analysis steps along the lowest imaginary mode both ways (largest atom 0.15 Å),
  re-optimizes both and keeps the lower energy, at most N times, a new Hessian after each. A clean minimum costs
  nothing extra. The result records it (`imag_found`, `imag_tightened`, `imag_steps`, `imag_energy_change_ev`,
  `imag_max_shift_A` in the vibrations dict, a note). On b03's 150 cases: 146 become minima (the energy falls by a
  median 0.15 kcal/mol; about 2 s per affected molecule); 4 near-free rotors stay at -22 to -36 cm-1.
- A run folder started before this resumes with the old behaviour (`freq_imag_retries` 0), as for the rank step.

### The ESP distribution on the molecular surface (2026-11-12)
- With `--cubes` the QM stage now keeps the whole distribution of the electrostatic potential over the 0.002 au
  density surface, not only Spartan's three polar areas (|V| > 75/100/125 kJ/mol), which stay:
  - 22 descriptors `qm_surf_esp_area_<lo>_<hi>_A2`: the surface area in each 25 kJ/mol interval of V from -250 to
    +250 kJ/mol (`m`/`p` = minus/plus: `qm_surf_esp_area_m025_p000_A2` is -25 <= V < 0), plus
    `qm_surf_esp_area_below_m250_A2` and `qm_surf_esp_area_above_p250_A2`; they add up to `qm_surf_area_A2`.
  - The finer profile, 5 kJ/mol intervals over the molecule's own range, in the summary
    (`qm.surface.esp_profile`) and as `<id>_esp_profile.csv` (lo_kjmol, hi_kjmol, area_A2), so another binning or
    the polar area at any multiple of 5 kJ/mol can be had later without the cubes (`chem.surfaces.profile_area`,
    `profile_polar_area`). The ESP analogue of a COSMO-RS σ-profile.

### Machine control: the split, each machine's budget, and pausing a machine, DIST-7 (2026-11-11)
- **Pause a machine** (the Machines dock's Pause / Resume per machine, `quantabench-cli worker pause [NAME]
  [--drain|--now]`, `worker resume`, `POST /v1/pause {"paused", "mode"}`): it takes no new molecules; `drain` (the
  default) lets the running ones finish, `now` stops them at their next checkpoint (they count as not started). A
  worker's batch then ends as `paused`, freeing its lanes, and goes on at resume. The pause is saved in the worker
  state folder (`pause.json`), so a restart stays paused; this machine's worker, a `grid build`'s local share and the
  GUI's own batch (which holds and goes on) all follow it. `grid machines` and the dock show it.
- **The work moves:** while `grid build` follows a grid, a paused machine's unstarted molecules go to the other
  machines by their rate (DIST-6 work sharing stays on). A worker that stops answering (a closed laptop) is away after
  `dist.away_after_s` (120 s): its unstarted molecules go to the others, after `dist.away_running_after_s` (600 s)
  the running ones too, and `--wait` stops waiting for it. When it answers again, or notices its own sleep (a
  wall-clock jump) or that its client stopped polling, it holds new molecules until the client says which are still
  its own (`POST /v1/jobs/<id>/sync`; the synced `grid.json` if the client is gone; at most `dist.rejoin_wait_s`).
  `GET /v1/jobs/<id>/molecules` gives the client each molecule's status; `release` now works on paused and queued
  jobs.
- **No double counting:** a molecule two machines finished is merged once, from the first to finish
  (`grid.json` `duplicates`); `timings` marks the dropped rows `duplicate=1` and prints the molecules per machine.
  `grid build --wait` and `grid status` end with the molecules each machine contributed, the moves and the
  measured rates.
- **The split:** `grid build --weights auto` (default) follows each machine's split setting (the dock's *Auto
  (measured speed)* / *Manual weight* / *Off*; `machines set NAME --split ... --weight W`); `measured` uses the
  molecules/hour of the last grid of the same protocol (`grid.json` `rates`, written at the end of each followed
  grid; paused or away machines are not recorded), else the benchmark; `equal`; `benchmark` (the old `auto`);
  `--exclude NAME`. The dock shows each machine's share of the current molecule list.
- **Processing budget per machine, also remote:** the dock's budget box and `quantabench-cli machines set NAME
  --priority --free-threads --max-cpu --lanes --[no-]pause-on-battery`. A running worker re-reads its values every
  15 s: the run priority is applied to its running lanes at once (`priority.apply_to_pids`), lanes and free threads
  with its next job. `pause_on_battery` (on by default) pauses a laptop's worker on battery and resumes it on power.
- **Settings are merged, not overwritten:** every save merges with the file as it is now (per-machine tables entry
  by entry, the rest key by key) and does not write when nothing changes (the GUI saves at every Run); per-machine
  values are stamped (`dist.machine_updated`) and a newer entry in a SyncThing conflict copy
  (`settings.sync-conflict-*.json`) is merged back on load. Workers never write settings.json: their state goes to
  `pause.json` and `Settings/machines/status/<name>.json`.
- **The Mac's worker** was found gone on 2026-11-09 because a restart ended its `nohup` process (sleep does not):
  docs/WORKERS.md now says to run it from the LaunchAgent and why.

### Density and polarizability estimates for every molecule, DESC-2 (2026-11-11)
- **`chem/estimates.py`**, in the descriptors step of every phase (graph or best conformer, milliseconds):
  `density_girolami_g_cm3` and `molar_volume_girolami_cm3` (Girolami's atom-volume rule), `density_packing_g_cm3`
  and `molar_volume_packing_cm3` (V_m = N_A V_vdW / k: the Bondi van der Waals volume of the best conformer and a
  packing coefficient from composition, H-bond donors and rings, fitted to 831 PubChem/HSDB liquid densities near
  25 °C), `polarizability_bosque_A3` (Bosque & Sales' composition model), `polarizability_crippen_A3`
  (Wildman-Crippen molar refractivity through Lorentz-Lorenz), `polarizability_d4_A3` (the D4 model with EEQ
  charges; only where PyTorch and tad-dftd4 or the dxtb fork are installed) and `parachor_sugden` (Sugden's group
  parachor, Quayle's table). A method with no parameters for an element of the molecule writes no value (an empty
  cell) and a note; the densities are not defined for ions.
- **Measured** (docs/VALIDATION.md, DESC-2): densities on 237 PubChem test liquids, 2.7 % mean error for the packing
  model and 5.3 % for Girolami; on 26 fluorinated liquids (≥ 3 F) 3.2 % and 6.4 %, on the 11 of them the fit
  never saw 3.8 % and 6.8 %. Polarizabilities against
  Lorentz-Lorenz values from PubChem's n_D and ρ: Crippen 1.7 %, Bosque-Sales 2.8 %, D4 5.2 % (696-726 liquids);
  against ωB97X-D/def2-SVPD (Psi4, 48 molecules, 14 of them PFAS): Bosque-Sales 4.2 % and Crippen 5.0 % (r 0.996 and
  0.998, PFAS included), D4 1.01 x DFT for organics but 1.11 x for PFAS, GFN1-/GFN2-xTB at 0.59 and 0.69 of DFT.
- **Both xTB polarizabilities in every run:** the properties step adds GFN2-xTB's (`tblite_gfn2_polarizability_*`,
  tblite, the run's solvent) when the refinement engine gave none, and computes the gas-phase GFN1-xTB properties
  with tblite where dxtb is not installed.
- **Methods guide:** a "Property estimates" family (Girolami and packing densities, Bosque-Sales, Crippen, D4 and
  xTB polarizabilities, Sugden parachor) with what each corrects and when to use it; docs/METHODS.md regenerated.
- **docs/SURFACE_TENSION.md:** the options for densities and polarizabilities (accuracy, cost, coverage,
  dependencies, license), which surface tension values to train on (reference tiers, the 20 → 25 °C correction,
  the gases), and where to get Jasper, ThermoML and DIPPR.

### The rank step: g-xTB + CPCM and AIMNet2 rank the conformers, RANK-2 (2026-11-11)
- **A rank step between the refinement and the descriptors** (`chem/rank.py`): every refined conformer gets a single
  point at the rank level at its refined geometry, and those energies set the lowest conformer, the window and the
  Boltzmann weights. Every descriptor taken from the lowest conformer or the ensemble follows it: `dipole_debye_best`
  and `_boltz`, the 3D `_best`/`_boltz` descriptors, `xtb_*`, the frequencies and the QM stage's `--qm-top`
  conformers. Before the step the refined conformers are pruned only as RMSD duplicates; the rank window (default
  30 kJ/mol) prunes after it.
- **`rank_method`** (BuildConfig, settings `build.rank_method`, `build --rank`, `grid build --rank`, worker job
  configs, the GUI's new Conformer ranking page): `auto` (the default), `refine` (the refinement's energies and
  `--window`, as before), `gxtb+cpcm` (g-xTB gas + ΔG_solv = E(GFN1-xTB CPCM) − E(GFN1-xTB gas), tblite's ddX, in the
  run's solvent), `gxtb`, `aimnet2`, `aimnet2+cpcm`, `qm` (the QM stage's level on every conformer). `auto` ranks by
  phase, not by fluorine content (RANK-1b): in a solvent `gxtb+cpcm`, else `aimnet2+cpcm`, else `refine`; in the gas
  phase `aimnet2` where its elements, charge and a closed shell allow, else `gxtb`, else `refine`. A level that cannot
  run for a molecule falls back with a note. `rank_window_kj` (`--rank-window`, shown in the display unit).
- **The outputs record it:** `summary.csv` `rank_method`, `rank_level`, `rank_energy_eV`, `rank_pick_changed`,
  `rank_n_pool`; `descriptors.csv` `rank_method`; per conformer `rank_energy_ev` and `refine_rel_kcal` (summary JSON,
  the SDF's `QB_RankEnergy_eV`/`QB_RefineRelEnergy_kcal`/`QB_RankLevel`, two new columns in the GUI's conformer
  table); `<id>_summary.json` `rank` with every pooled conformer's refinement and rank energies, their terms, the
  pick and the engines; the phase note `rank: g-xTB + CPCM(GFN1, water)`. `energy_ev` stays the refinement's;
  `rel_kcal` and the weights follow the rank level.
- **g-xTB** (`chem/eng_gxtb.py`): the official grimme-lab/g-xtb v2.0.1 build of xtb (`xtb --gxtb --chrg Q
  --norestart`, `--uhf` only for open shells), its own program entry `gxtb` (a plain `xtb` is never taken for it),
  found through this machine's path in the settings or `QUANTABENCH_GXTB`. Every molecule's `rank` record and the
  run's telemetry name the build (release, platform, SHA-256 of the executable); `doctor` shows it. A binary that
  does not start counts as missing. A rank level only, not a refinement engine. GPL-3.0: run as a separate program.
- **Installers:** `--with-engines` / `-WithEngines` download the platform's g-xTB v2.0.1 build from
  `scripts/fork-packages.json` ("programs" > "gxtb", pinned by URL and SHA-256) and set it as this machine's path.
  The macOS arm64 build ships without `libgcc_s.1.1.dylib`; `install_macos.sh` copies conda-forge's from `qb-tools`.
- **Telemetry:** a `rank` stage (wall and CPU time, the engine with the g-xTB build); `telemetry_runs` records the
  g-xTB build and the rank settings.
- **Cache:** g-xTB and tblite single points are cached per conformer in the per-machine cache folder
  (`rank_singlepoints.sqlite`), so a molecule computed again after an interruption does not redo them.
- **Resume:** `rank_method` and `rank_window_kj` are science settings (a change is refused unless forced). A run
  folder from before RANK-2 resumes as `refine`, unchanged: a stored config without `rank_method` ran at the
  refinement level (`jobs.LEGACY_DEFAULTS`, `jobs.stored_science`).
- **The review** (ADV-1) judges the level that ranks each molecule. In water the rank step (`rank_method auto`)
  replaces the DFT re-rank it recommended, and AIMNet2 single points (the rank step) the switch of the refinement
  method in the gas phase; where no rank engine is installed it says so and keeps the DFT re-rank. The presets set
  `--rank auto`: Recommended in water no longer runs DFT; Careful checks the rank step's lowest 8 with
  ωB97X-D/def2-SVPD. New findings for g-xTB + CPCM, AIMNet2 + CPCM, g-xTB and a narrow rank window.
- Checked: RANK-1b's 136 pools and RANK-1's 8 holdout ions through the rank code path give the benchmark's picks in
  196 of 196 molecule-levels, with relative energies equal to 2e-9 kJ/mol (docs/VALIDATION.md, "RANK-2 integration
  check"). In real builds on NOVA the step changed the pick in 6 of 8 PFAS and other ions in water and 6 of 10
  neutrals in the gas phase.
- `grid build --rank auto` warns that a machine without the rank engines falls back to another level.
- The methods guide's g-xTB and conformer-ranking entries, docs/METHODS.md, docs/ENGINES.md ("The rank step",
  "g-xTB"), docs/CHOOSING_METHODS.md.

### PubChem target tables, for the staged QSAR batches (QSAR-1, 2026-11-11)
- **`quantabench-cli pubchem targets RECORDS --out TABLE`** (`chem/pubchem_targets.py`): one value per molecule
  from the long-form records of `pubchem enrich --records`, e.g. the pure-liquid surface tension at 25 °C. Records
  of the wanted condition class with a stated temperature in a window (`--window`, default 20-30 °C), moved to the
  target with the molecule's own dσ/dT when its records give a plausible one, else -0.1 mN/m per K; one value per
  reference, references tiered (compilations such as the CRC Handbook and Riddick first, the U.S. Coast Guard's
  CHRIS manual last) and the median of the best tier. Records to review, estimated values and records without a
  temperature are left out and counted; surface tensions printed in N/m for mN/m are repaired and flagged. The raw
  value, its temperature, the correction and the spread over all references stay in the row. `--heading Density`
  makes a density table the same way.
- docs/ENGINES.md (PubChem) and docs/SURFACE_TENSION.md (Data assembly); BACKLOG QSAR-1 is now the staged-batch
  plan.

### Run priority and free threads, PRIO-1 (2026-11-09)
- **Runs give way to what you do on the PC.** A new run priority per machine: `normal`, `low` (the new default)
  or `background`. `low` means below-normal CPU and low I/O priority on Windows (nice 10 and ionice best-effort 7
  on Linux, nice 10 and the utility disk I/O policy on macOS). `background` means idle CPU and very low I/O
  priority (nice 19 and the idle I/O class; the throttle policy on macOS), and at least 25 % of the threads left
  free.
  - It applies to the lane processes, a CLI `build`, `grid build` or `worker` process, and in the GUI only the
    build thread of in-process work (never the window).
  - It also applies to every program QuantaBench starts (xtb, CREST, MOPAC, Psi4), with what those start.
    Through WSL, `nice`/`ionice` go into the wrapper, since Windows priorities do not reach the Linux VM.
  - A priority is only ever lowered.
  - Before, runs used normal priority, unless started from a lowered shell, and normal I/O priority.
- **Free threads per machine** (default 1, the old "all threads but one", so every lane plan stays as it was):
  the automatic thread budget, the `auto` lane plan and the threads dynamic CREST threads borrow leave them free.
  An explicit thread count or lane spec is used as given.
- Settings > Compute: "Run priority" and "Leave free". `quantabench-cli settings --priority/--free-threads
  [--machine NAME]`; `--priority`/`--free-threads` on `build`, `grid build`, `worker` and `lanes`. `doctor`
  prints this machine's values, and run timings record them in `telemetry_runs.config_json` (no schema change).
- Both are stored per machine in the shared settings (`resources.machine_priority`,
  `resources.machine_free_threads`), like the CREST path and the lanes; QuantaBench 0.3.0.dev3 and older drops
  them when it saves the settings.
- Checked on NOVA: programs at `low` and `background` and their children get the priority class and the I/O
  priority. docs/ENGINES.md, "Run priority and free threads".
- `quantabench-cli lanes` without a spec shows this machine's lane spec (`Settings.lanes_for`), not the shared
  default.

### Live machine metrics, MON-1 (2026-11-09)
- **What the machine is doing while a job runs** (`quantabench/sensors.py`): CPU utilization (total and per logical
  CPU), the effective CPU clock as Task Manager shows it (average, min-max over the cores) and the base clock, CPU
  temperature and package power, GPU utilization, temperature, clock, power and VRAM, RAM, fans where cheap.
  Windows: the performance counters through ctypes (no new dependency) and LibreHardwareMonitor's web server for
  temperatures and power; macOS: macmon (`brew install macmon`); Linux: cpufreq, hwmon, RAPL, amdgpu sysfs and
  nvidia-smi; WSL2 says the sensors are on the Windows side. A read never raises and takes about a second at most;
  whatever is missing says why and how to enable it.
- **GUI: a machine monitor strip** in the Pipeline dock, above the Run buttons: this computer and every worker, with
  CPU %, effective clock, temperatures, GPU % and VRAM, RAM, 10-minute sparklines and colours by threshold. It
  samples on background threads and folds to one line. The Machines dock's worker status shows each worker's load
  and temperature too.
- **`quantabench-cli monitor [--interval 2] [--count N]`** prints one line per sample (`--per-cpu`, `--json`,
  `--worker NAME`); `monitor --set lhm_url=... [--machine NAME]` stores this machine's sensor settings.
  `quantabench-cli doctor` has a Sensors section.
- **Workers** answer `GET /v1/sensors` and add the latest sample to `/v1/info`; older workers answer 404 and are shown
  as "older worker without sensors".
- **Run timings:** the job table's schema is 5: the 60 s load samples also hold the effective clock, CPU temperature
  and power, and GPU utilization, temperature and power (older tables get the columns on their next run).
  `quantabench-cli timings` prints mean/max temperature and clock per run and flags likely thermal throttling
  (a clock well below the run's typical clock under the same load, at >= 85 °C).
- **Settings:** `sensors.enabled`, `sensors.interval_s` and `sensors.lhm_url`, each also per machine in
  `sensors.machine` (the file is shared by SyncThing); the defaults need no configuration. docs/ENGINES.md, Live
  machine metrics.

### The review before a run: a model advisor, ADV-1 (2026-11-09)
- **QuantaBench reviews the molecules before a run** and recommends methods, saying what each one is for and what
  it corrects (`chem/advisor.py`; user guide docs/CHOOSING_METHODS.md).
  - The batch: elements, charge class at the run's pH, PFAS (OECD 2021) and chain length, functional groups,
    sizes, salts, stereo, radicals, isotopes, metals. Every count lists its molecules.
  - Which methods, basis sets and CREST levels can take the batch (elements, charge, spin), and where each step
    runs in a solvent run.
  - Findings tied to QuantaBench's measured results. Examples:
    - GFN2-xTB ranks PFAS ions no better than a random pick (RANK-1, RANK-1b); the finding recommends a DFT
      re-rank in PCM on a wide pool until the rank step (RANK-2) lands.
    - AIMNet2 is a closed-shell model, which method `auto` does not check.
    - A functional without dispersion, or a basis without diffuse functions for anions (def2-SVP favours compact
      PFAS-ion conformers by 15 kJ/mol per Å).
    - Ions in the gas phase; salts; undefined stereocentres; a proton on a sulfonamide N.
  - Three presets (Fast screen, Recommended, Careful) with estimated times on this machine.
- **App:** the Review window opens when you press Run (Run anyway starts the build; Build > Review before each run
  turns it off) and from Review... or F6. It has the batch with click-through to the molecules, Apply and Explain
  per finding, and the presets.
- **CLI:** `quantabench-cli review` (build's inputs and options, `--phase gas|water`, `--json`). `build` and
  `grid build` print a short review first; `--no-review` skips it.
- **Methods guide:**
  - New entries for g-xTB, r2SCAN-3c and the -3c composites, basis-set superposition error and gCP, conformer
    ranking, and protonation.
  - "Corrects / Use for / Avoid for" lines on 35 entries.
  - docs/METHODS.md gains the element, charge and spin coverage table (`method_guide.COVERAGE`, with sources).
- **Faster protonation:** the rule table's SMARTS are compiled once and its distance matrix is built only when a
  base sits near an N+. The results are unchanged.

### Conformer ranking benchmark, RANK-1b (2026-11-08)
- docs/VALIDATION.md, "RANK-1b": RANK-1's cheap rankers on Brandon's non-fluorinated handbook molecules (12
  hydrocarbons, 28 polar organics, 8 ions) and a matched control of 20 neutral PFAS. All went through one protocol
  against r2SCAN-3c in the gas phase and in PCM water; RANK-1's 13 PFAS ions are the ion context.
  - GFN2-xTB's weak ranking of neutrals is about polarity, not fluorine: in water, 16 of 28 polar organics and 14 of
    20 PFAS within 4 kJ/mol.
  - Its catastrophic ion failure is fluorine's: 0 of 13 PFAS ions, against about a random pick on non-fluorinated
    ions.
  - g-xTB, AIMNet2 and g-xTB + GFN1-xTB CPCM generalize to non-fluorinated neutrals.
  - r2SCAN-3c agrees with ωB97X-D/def2-TZVP. def2-SVP's lean toward compact conformers also appears without
    fluorine, at half the PFAS anions' size.
- No code change. BACKLOG RANK-2 now picks the rank level by phase: g-xTB + CPCM in water, AIMNet2 in the gas phase.

### Fixes (2026-11-08)
- `tests/test_lanes.py::test_speed_aware_fast_lane_crash_releases_the_slow_lane` failed about one run in three:
  the fast lane crashed the moment it took TFE, sometimes before the slow lane had started, so the slow lane was
  never held back. The crash test hook takes a delay (`QUANTABENCH_TEST_LANE_CRASH=TFE@2`) and the test uses it.
- Run timings on Windows (Python 3.12): the load sampler's stop event was called `_stop`, which hides
  `threading.Thread._stop()`, so stopping it raised and the run row was never closed (no end time, an interrupted
  molecule left as "running"). Renamed.
- `tests/test_sensors.py::test_linux_sysfs_provider` skips on Windows: its fake sysfs needs names like `intel-rapl:0`,
  which Windows forbids.
- A program's CPU time on Windows counts the processes it starts (a job object, as `getrusage` does on POSIX):
  a venv's `python.exe` is a launcher, and its own time was all that was recorded.

### Run timings (2026-11-08)
- **Every batch records its timings in its job table** (`chem/telemetry.py`, SCHED-1 step 1). Per run: versions,
  machine, CPU, cores, RAM, thread budget, lane plan, dynamic threads, engines and their versions, the settings
  that change the cost, and the machine's load every 60 s. Per molecule attempt and stage: lane, threads planned,
  borrowed and given (CREST's `-T`), wall and CPU time (also of CREST, xtb and Psi4), peak memory, queue wait,
  outcome, conformers in and out, and the molecule's size, rotatable bonds, rings, charge and F/O/S/N counts.
  Lanes send their numbers with the result; a lane that dies or is killed leaves a partial row. Always on, about
  0.3 % of a run; it never fails a molecule or a run. Old run folders get the tables on their next run.
- **`quantabench-cli timings [PATHS]`** collects them from run folders and grid parts into one CSV, with a
  summary per machine, stage and engine. **`timings --fit`** refits the CREST cost model (time ~ atoms^p,
  speedup threads^q) per machine and CREST level, and saves it to `<settings>/machines/cost-models/` when there
  are at least 30 molecules over 2 thread widths; the dynamic CREST threads then use that machine's constants.
  docs/ENGINES.md, Run timings.

### Conformer ranking benchmark, RANK-1 (2026-11-08)
- docs/VALIDATION.md, "RANK-1": which cheap method ranks PFAS conformers as DFT does, on 17 ions in water
  (ωB97X-D/def2-SVP PCM, a holdout of 8 included) and the Essential 76 neutrals in the gas phase (Spartan's
  ωB97X-D/6-31G*). A reference check with r2SCAN-3c and ωB97X-D/def2-SVPD finds the def2-SVP reference
  favours compact conformers; g-xTB gas + GFN1-xTB CPCM ΔG_solv is the recommended cheap ranker for ions. No code
  change; the pipeline's rank step is BACKLOG RANK-2.
- Methods guide: GFN2-xTB, ALPB, PCM, GFN1-xTB, AIMNet2, PM7, PM6-D3H4X, GFN-FF, MMFF94s, B3LYP-D3(BJ), r2SCAN,
  HF, def2-SVP and def2-SVPD say what the benchmark found. GFN2-xTB ranks PFAS anions' conformers no better than
  chance and overbinds short anionic O···F contacts. GFN1-xTB's CPCM solvation differences track IEF-PCM's.
  AIMNet2 is the best cheap gas-phase ranker for the neutrals. def2-SVP favours compact anion conformers. The
  entries also give r2SCAN-3c, HF-3c and def2-SVPD costs in Psi4.

### Tests never reach the real settings (2026-11-08)
- A GUI test called `monkeypatch.undo()`, which also undid the per-test settings folder; its window then saved
  on close into the folder `settings_location.json` points at. On the MacBook (2026-11-07 15:10) that overwrote
  the shared settings.json: `output_dir` became a pytest folder and the per-machine engine paths were dropped.
  conftest now points the settings, window state and cache at a session folder before any test runs, so nothing
  in a test run can fall back to the real one; the test uses `MonkeyPatch.context()`. The settings were put back
  (output_dir `~/SyncThing/QuantaBench/Runs`, the MacBook's CREST path); the overwritten file is kept as
  `settings.json.bak-test-overwrite-20261108-011301`.

### Dynamic CREST threads (2026-11-07)
- **A CREST search may use idle lanes' threads** (`dist/threads.py`, `dist/lanes.py`). The parent of a parallel
  run owns the CPU lanes' threads and sets each CREST call's `-T` when it starts. A molecule that would set the
  end of the batch on its own gets whole lanes' worth more; a run with fewer molecules than lanes lends the
  unused lanes' threads, and a run of one molecule gives its search all of them. The threads go back when the
  search ends; refinement and properties stay on the lane's threads. The threads in use never exceed the plan's.
  The log shows each loan ("54950-05-9: CREST with 8 threads (borrowed 4 from idle lanes)").
- The rule was chosen with a batch simulation on the measured CREST times: no change for the Mac's 3 x 4 on the
  20 PFAS or for large batches, NOVA's 11 x 2 on the 20 PFAS 1,244 -> 819 s. On the Mac (GFN-FF gas): the 20 PFAS
  842 s with nothing widened; two large PFAS 585 and 555 s instead of 1,309 and 1,096 s (CREST's sampling is
  part of that).
- On by default. `build --no-dynamic-threads`, Settings > Compute "Dynamic CREST threads"
  (`resources.dynamic_threads`) turn it off; `lanes` says when it applies.

### Methods guide (2026-11-07)
- **What the methods mean** (`chem/method_guide.py`): one entry for every engine (GFN2-xTB once, with how dxtb,
  tblite and the xtb program differ), CREST level, functional, basis set, solvent model and optimizer, plus
  Spartan's B3LYP/6-311+G** (C-PCM), ωB97X-D/6-31G* and PM6, and the concepts behind them (dispersion
  corrections and why plain B3LYP favours extended chains, basis-set names, implicit solvation, single point vs
  optimization, Boltzmann weights, xTB HOMO/LUMO as trend descriptors). Each says what it is, what it is good and
  bad at, the cost (QuantaBench's own timings where there are some), dispersion, solvation and 1-3 references
  (checked against Crossref). 49 entries.
- **GUI:** the pipeline pages show the entry's one-line summary under Method, Solvent (the model the chosen engine
  uses), Optimizer, Search, CREST level, Functional and Basis set, with a "More…" link; the items' tooltips carry
  the summaries. Method keeps its availability and gas-phase warnings. Help > Methods guide... (F1) lists every
  entry by family, with a search box and links between entries.
- **CLI:** `quantabench-cli methods` lists the entries, `methods NAME` shows one (case, hyphens and ω/w do not
  matter: `methods wB97X-D`, `methods gfn2`, `methods 6-31G(d)`; `methods wb97x-d/def2-svp` shows both).
- **docs/METHODS.md** is written from the same data by `scripts/write_methods_doc.py`; a test fails when it is out
  of date. A test also fails when an engine, CREST level, functional or basis set in the GUI has no entry.
- **The DFT conformer test** (docs/VALIDATION.md, VAL-3) is now in the B3LYP, dispersion and GFN2-xTB entries.
  For PFAS ions in water, the missing dispersion in Spartan's B3LYP is a small part of the disagreement with
  GFN2/ALPB; ωB97X-D and B3LYP-D3(BJ) also put Spartan's pick below GFN2's in most ions, and GFN2's ranking is the
  outlier.

### After the merge (2026-11-07)
- **One CDX reader.** `.cdx` files and ChemDraw's clipboard data go through the same converter
  (`chem.interchange.cdx_to_cdxml`), which was checked against files ChemDraw wrote. The file reader's own copy
  expected a 28-byte header before the document object, so it would have failed on real ChemDraw files.
- **The Mac's CREST build is released:** `brandonbtm29/crest` release `qb-v3.0.2-macos1` (the CI tarball,
  unchanged). `scripts/fork-packages.json` points at it, so `install_macos.sh --with-engines` installs it.
  The Accelerate bug is reported to conda-forge (crest-feedstock#22) and on crest-lab/crest#412.

### CREST lanes on Apple Silicon, largest molecules first (2026-11-07)
- **CREST lanes on Apple Silicon** (`dist/lanes.py`, `resources.py`): `--lanes auto` for a CREST run gives
  4 threads per lane, one lane per 4 performance cores (`sysctl hw.perflevel0.physicalcpu`; the physical cores
  when that cannot be read). The M3 Max gets 3 x 4 instead of 6 x 2. On that machine 3 x 4 took 773 s and 6 x 2
  3,108 s for 20 PFAS with GFN-FF in the gas phase. Other machines keep 2-thread lanes on all threads (NOVA:
  11 x 2). `doctor` shows the performance cores ("14 cores (10 performance)").
- **Largest molecules first** (`dist/lanes.py`): parallel runs hand out the largest molecules first (atoms
  with hydrogens), as grid shares already did. A batch no longer ends with one large molecule on 2 threads while
  the other lanes are idle. Molecules that crashed a lane still go last. Results stay in input order.
- **`settings --engine-path NAME=PATH --machine`**: `--machine` with no name means this machine (before, it
  was an error).

### 3D model kit core (UI-7, 2026-11-07)
- **Edit model** (`chem/edit.py`, no GUI yet): an RDKit molecule with 3D coordinates where every change is an
  undoable command. Pick an atom type (C sp3 / sp2 / sp, N, O, P, S, Si, H, halogens, =O, =S, ≡N, as in
  Spartan's Organic tab) or a fragment and attach it at a free valence: it goes at its ideal bond length, and
  its open valences get hydrogens. Also replace an atom, delete, bond order, formal charge, make/break bond,
  set distance / angle / dihedral (the smaller side moves; ring bonds are refused), Minimize (MMFF94, UFF
  when MMFF has no parameters, frozen atoms) and a local relax near the last edit. SMILES and MOL blocks in
  and out; `to_input()` hands the molecule to the pipeline.
- **Fragment library** (`chem/fragments.py`): groups, rings (chair cyclohexane), substituents and PFAS
  pieces (CF3, `(CF2)n` and `CnF2n+1` builders, ether O, fluorotelomer CH2CH2, PFCA/PFSA heads). Chains
  start extended (anti).
- Tests build PFOA, PFOS, GenX and 6:2 FTS with kit operations only: the SMILES match, and after Minimize
  every bond is within 0.03 Å of an MMFF94 reference.

### Import from many formats, with Spartan-style import options (UI-4, 2026-11-07)
- **More formats** (`chem/formats.py`, through `inputs.read_file`, so File > Open and `--file` agree): MOL2 (several
  molecules per file), Marvin MRV, ChemDraw CDXML and binary CDX, CML, InChI lists (`.inchi`), CIF of molecular
  crystals (molecules made whole across the cell edges, one entry per unique molecule), Gaussian `.gjf`/`.com` and
  ORCA `.inp` inputs (bonds from the geometry with the file's charge; a radical keeps single bonds; a geometry with
  several molecules is split). SDF V3000 already worked and now has a test. RDKit 2026.09's Python ChemDraw reader
  takes every file as CDXML, so binary CDX is turned into CDXML first (`formats.cdx_to_cdxml`).
- **Open Babel fallback:** another extension is converted to SDF by the user's `obabel` (Settings > Engines, PATH,
  conda environments) in a temporary folder. Open Babel is GPL-2.0: it is only run, never imported or bundled.
  Each molecule records its reader (`meta["qb_reader"]`) and the log says it.
- **The Molecules box** also takes InChI (parsed locally) and InChIKeys (PubChem, like names).
- **Import options** (`chem/import_options.py`, the Import Options dialog of File > Open, Settings > Import,
  `Settings.import_`): convert 2D structures to 3D, grow hydrogens, keep the file's 3D coordinates as the first
  conformers of the build (the pipeline seeds RDKit's conformer search with them), post SD fields and table columns
  to the molecule list and `summary.csv` (numbers parsed), id and name fields, filters (water, HCl and small
  inorganic molecules, counter-ions, largest fragment; metal fragments are left for the build), atom order (read or
  standard), several structures of one constitution as molecules or conformers. What the import removed is in the
  molecule's notes and in `removed_fragments`. The dialog appears when a file has choices; "Don't ask again"
  keeps them.
- **CLI:** `--keep-coords`, `--grow-h/--no-grow-h`, `--convert-2d/--no-convert-2d`, `--filter`, `--label-order`,
  `--same-structures`, `--post-fields`, `--name-field` and `--import-options key=value,...` for `--file` inputs;
  the defaults are the stored choices.
- File > Open hands `.spartan`, `.log` and `.out` files to Import calculation output.
- SD properties keep their text as written ("007" was read as 7).

### Paste from ChemDraw, Copy as SMILES / MOL / CDXML, drag and drop (UI-4, 2026-11-07)
- **Edit > Paste (Ctrl+V)** adds the structures on the clipboard to the Molecules list (`gui.clipboard`). It
  tries ChemDraw's own format (binary CDX: Windows "ChemDraw Interchange Format", macOS
  `com.cambridgesoft.chemdraw.cdx`), then CDXML, an MDL molfile (`chemical/x-mdl-molfile`, Windows "MDLCT"),
  copied files, and text. The log says which format it used. In a text field, Ctrl+V pastes text as before;
  in the Molecules box it adds molecules only for a structure format or several lines of text.
- **Text formats** (`chem.interchange.parse_structure_text`): MOL blocks (V2000, V3000), SD text with several
  records, CDXML, Marvin MRV, else lines of SMILES, names and CAS numbers (`inputs.parse_text`). Parsing runs
  off the GUI thread, since names go to PubChem.
- **Binary CDX** (`parse_cdx_bytes`): RDKit 2026.09's Python wrapper reads CDX only as CDXML, so
  `cdx_to_cdxml` translates the CDX objects that describe molecules (fragments, nodes, bonds, labels,
  nicknames such as CF3) and RDKit's CDXML reader does the chemistry. A newer RDKit that reads CDX bytes is
  used first. Checked on ChemDraw-written files from RDKit's test data (same SMILES and stereo as their
  CDXML). The file header is optional, since clipboard data may lack it.
- **ChemDraw salts:** when a drawing's charged fragments add up to zero (K+ and PFOS-), they become one
  molecule; neutral fragments stay separate molecules.
- **Edit > Copy as SMILES / MOL / CDXML** puts the selected molecules (as entered) on the clipboard: SMILES
  lines, a 2D MOL block (V3000 only when needed; SD text for several) or CDXML. Plain text is always there,
  for ChemDraw's Edit > Paste Special.
- **Drag and drop:** structure files and tables open as with File > Open; calculation outputs and Spartan
  documents go through Import calculation output; a folder opens as a run folder; dropped text is read like
  a paste.
- Needs a hand check: copying from real ChemDraw on Windows and macOS (only Brandon has ChemDraw).

### 2D sketcher: Schrödinger's Sketcher in a Sketch dock (UI-6, 2026-11-07)
- **Sketch dock** (Build > Sketch a molecule..., View > Sketch): Schrödinger's 2D Sketcher (BSD-3-Clause,
  WebAssembly build of release v2026-4) in Qt WebEngine. New, Edit selected, Add to list, Replace selected
  (keeps the id, name and place in the list) and Insert fragment (CF3, (CF2)n, COOH, COO-, SO3H, SO3-, SO2NH2,
  CH2CH2, ether O, phosphate). The drawing goes over as MOL V3000 and is checked with RDKit first
  (`chem/sketch.py`): valence, aromatic rings that cannot be kekulized and query atoms are reported in words.
- **Optional:** `pip install -e ".[editor]"` adds Qt WebEngine (the `PySide6` package). Without it the dock says
  what to install and nothing else changes. `QUANTABENCH_NO_SKETCHER=1` keeps Qt WebEngine from loading.
- **The editor's files are downloaded on first use** (17.5 MB, `sketcher_assets.py`) into the programs folder,
  checked against a pinned sha256, unpacked, with Schrödinger's licence next to them; nothing large is in the
  repository. `quantabench-cli sketcher --install [--from-zip FILE]` fetches them ahead of time or offline;
  `--use DIR` or `QUANTABENCH_SKETCHER_DIR` points at a self-built version. `doctor` shows the state.
- **How the page is served** (`gui/sketcher_web.py`, `gui/sketcher_view.py`): a custom `qb-sketcher://` scheme
  registered before the QApplication (`gui/app.py`), no local server. The view draws with Qt Quick's software
  renderer, because Qt's GPU composition and VTK's OpenGL 3D view spoiled each other under Linux/Mesa
  (`QUANTABENCH_SKETCHER_GPU=1` turns it back on to try).

### CREST on the Mac, loud CREST failures, engine paths per machine (2026-11-06)
- **Why the Mac's CREST failed.** conda-forge's osx-arm64 `crest 3.0.2` links Apple's Accelerate, whose classic
  BLAS returns a double from `sdot` (f2c convention). ANCOPT's Davidson step (19+ atoms) then fails, so every
  initial optimization of 19+ atoms stops ("Initial geometry optimization failed!"). Separately, libgfortran 16
  stops `crest --opt` at ANCOPT's first printout (a format without a comma; Linux too). Details in
  docs/ENGINES.md (CREST) and the fork's `docs/MACOS.md`.
- **The fork's macOS build** (`brandonbtm29/crest`, branch `qb/macos`, workflow `qb-macos`): conda-forge's
  compilers with OpenBLAS instead of Accelerate (`-DBLA_VENDOR=OpenBLAS`), one OpenMP runtime (libomp),
  packaged as a folder that runs from anywhere (`crest-macos-arm64.tar.gz`: `bin/crest` + `lib/`).
- **CREST failures are loud** (`chem/pipeline.py`, `chem/crest.py`, `chem/jobs.py`):
  - each molecule: a note with CREST's own reason ("Initial geometry optimization failed! (exit code 1; job
    files in ...)"), a WARNING log line, and `summary.csv`'s new `conformer_fallback` column;
  - each batch: "WARNING: CREST failed for N of M molecule(s) ..." at the end, the count in `status`, in the
    GUI's run summary and on its status bar;
  - `--crest-fallback fail` (GUI: untick "If CREST fails, use RDKit's conformers") fails those molecules
    instead of building them with RDKit's conformers;
  - a CREST that exits 0 after a stage in which no optimization succeeded ("0 of N structures successfully
    optimized", crest-lab/crest#412) counts as failed;
  - `doctor` and the build log name a CREST linked to Accelerate on macOS.
- **Engine paths per machine** (`engines.machine_paths`, like `dist.machine_lanes`): the shared settings file can
  hold NOVA's `wsl:` CREST and the Mac's own. Settings > Engines saves for this machine only;
  `quantabench-cli settings --engine-path crest=PATH [--machine NAME]` sets it from a script. Older versions drop
  `machine_paths` when they save the settings: update every machine sharing the file first.
- **Saving keeps the other machines' entries** (`settings.save`): the per-machine tables (`engines.machine_paths`,
  `dist.machine_lanes`) are merged with the file as it is now, so a GUI that stayed open for hours (it saves at
  every Run) no longer writes back its old copy of another machine's CREST path or lanes.
- **Proposed install of the Mac build** (not released): `scripts/fork-packages.json` > `macos` > `programs` >
  `crest` (url, sha256, exe). `install_macos.sh --with-engines` downloads it, checks the sha256, unpacks it into
  `~/Library/Application Support/QuantaBench/programs` and sets the Mac's CREST path. The url stays empty
  until a fork release exists.

### Comparing conformer sets: `conformers compare` (2026-11-06)
- **`quantabench-cli conformers compare --a ... --b ... --out DIR`** (`chem/conformer_compare.py`): which
  conformers two programs found for the same molecules. Either side can be Spartan documents, a Spartan import
  (`import --conformers all --table --sdf`) or QuantaBench run folders, so CREST runs compare the same way.
  - Molecules join by constitution (`io_spartan.constitution_key`).
  - Two conformers are the same when their symmetric heavy-atom RMSD is at most 0.5 Å, their backbone and polar-H
    torsions agree within 60° and every stereocentre keeps its configuration. Mirror images count as the same.
  - PFAS symmetry is factored instead of enumerated. Terminal sibling atoms (the F of CF3/CF2, the O of
    SO3-/COO-) are assigned per group after aligning a reduced skeleton, so PFOS (4,608 automorphisms) costs
    1.6 ms per pair. Up to 2,000 automorphisms the full group is enumerated exactly, which equals RDKit's
    `GetBestRMS`.
- **Reports** (`molecules.csv`, `levels.csv`, `levels_summary.csv`, `conformers.csv`, `summary.csv`, `report.md`):
  - how many of A's conformers within 5 and 10 kJ/mol of A's minimum B found, and the reverse;
  - whether the two lowest are the same, with their RMSD and torsion differences;
  - per level: B3LYP, MMFF, QuantaBench's and the common one.
- **Search or ranking** (`--rescore gas,water --relax water`):
  - Every conformer of both sides gets GFN2-xTB single points (tblite, gas and ALPB) and a GFN2 optimization.
    The results are cached in `rescore_cache.jsonl`.
  - An optimization that changes the bonds (a proton moves) is set aside.
  - Each molecule then gets a cause: same best, ranking (B has A's best, or the common level would not prefer
    it), or search (B missed a conformer that its own level puts lower).
  - The dipoles of each side's lowest conformer split the dipole gap into method and conformer.
- **C3A/C3B water ions of the 2026-11-06 validation** (61; `Runs/conformers-compare-2026-11-06`):
  - Only 6 of the 61 lowest conformers are the same. QuantaBench has Spartan's lowest (B3LYP/C-PCM over its
    DFT_Top5) for 10.
  - The conformer explains 88 % of the 4.35 D dipole gap. GFN2/ALPB at Spartan's lowest geometry is 0.52 D from
    B3LYP; QuantaBench's own lowest is 4.35 D away.
  - Cause of the 55 differences: ranking 38, search 16, one tautomerizes at GFN2. GFN2 ranks Spartan's B3LYP-lowest
    11th of about 33 (median).
  - QuantaBench's search is still short at its own level: in 49 molecules Spartan's MMFF pool holds a conformer
    that GFN2 puts lower, on average 14.8 kJ/mol below QuantaBench's lowest.
  - That conformer has an even lower dipole: 6.2 D MAE from Spartan. A better GFN2 search alone would widen the
    gap, not close it.
- **Essential 76 in the gas phase** (63; Spartan's Est. DFT conformers against a new QuantaBench gas run):
  - 8 of the 63 lowest conformers are the same. Cause: search 35, ranking 20.
  - QuantaBench's lowest is 6.2 kJ/mol above GFN2's lowest of both sets (mean).
  - The 1.28 D dipole gap is 37 % conformer; the rest is the method (GFN2 is +0.76 D above ωB97X-D).
  - Spartan's TFA and PFPrA documents hold one anti-acid conformer each, 17-19 kJ/mol above syn at GFN2.
- `io_spartan.constitution_key` and `perceived_bonds`; `structure_key` uses them.

### Work sharing between machines in grid builds (2026-11-06)
- **A machine that finishes early takes the others' unstarted molecules.** While `grid build` follows a
  grid, it measures each machine's rate and moves molecules a busy machine has not started to an idle one.
  It moves only when the busy machine's part ends at least 60 s sooner, after a 30 s start-up charge.
  - The busy batch hands them over between molecules, from the end of its queue, and marks them `moved` in
    its job table, so no molecule runs twice. The idle machine runs them as a new share of the grid
    (`parent`, `ids` in `grid.json`).
  - On by default; `--no-share-work` keeps the split as it is. New worker endpoint:
    `POST /v1/jobs/<id>/release`.
  - In the 2026-11-05 water run, the MacBook had been idle for 8 of 21 minutes.
- **Largest molecules first:** each share's list is sorted by size before it is submitted.
- `jobs.Handoff` and the `moved` status: `RunStatus.total` leaves moved molecules out, and a resume does not
  run them.
- **Fix:** the hardware probe (`resources.detect_isolated`) ignored a cancel when its child process finished
  within the first 0.25 s poll. This made CI red on Linux and Windows.

### macOS: first full test run on the MacBook (2026-11-06)
- **Spectra panel under NumPy 2.** Selecting a mode passed a `numpy.bool` to Qt (`marker.setVisible(f > 0)`),
  which PySide6 refuses with NumPy 2 ("'numpy.bool' object cannot be interpreted as an integer"): every mode
  selection raised, so the info line stayed empty and no mode could be selected or animated. NOVA's NumPy 1.26
  hid it. A new test needs no cclib, so the core CI install runs it.
- **GUI tests run on macOS.** They were skipped wherever `$DISPLAY` was unset, which on a Mac is always: CI's
  macOS job and the MacBook never ran one. `conftest.HAS_DISPLAY` counts Windows and macOS as having a display.
- **CI on `macos-15`.** GitHub no longer starts `macos-14` jobs ("The configuration 'macos-14-arm64-us-default'
  is not supported"); every macOS job since 2026-11-05 20:24 UTC was cancelled in 2 s.
- **`detect_isolated(cancel=...)` with the event already set** returns None at once. Without torch (CI's core
  install) the child finished inside the first 0.25 s wait and its report came back instead, which failed
  `test_detect_isolated_runs_in_a_child_and_can_be_cancelled` on Ubuntu and Windows since 2026-11-05.
- **LAMMPS on macOS:** the `md` extra installs `lammps[mpi]` there. PyPI's macOS wheel links MPICH's
  `libmpi.12.dylib` without shipping it, so running LAMMPS failed with "Library not loaded"; the `mpi` extra
  brings PyPI's `mpich` wheel into the venv's `lib/`, where liblammps looks for it.
- **Conda environments outside `~/miniforge3`:** external programs are also found in Homebrew's Miniforge cask
  (which `install_macos.sh` uses when it finds it) and in any environment listed in `~/.conda/environments.txt`.
- `test_machine_name` no longer depends on the Mac it runs on (the Bonjour name came first there).

### Energies in kJ/mol: Settings > Display (2026-11-06)
- **The GUI shows energies in kJ/mol by default**, as Brandon's Spartan does. **Settings > Display** (Spartan's
  Preferences > Document) sets:
  - **Energies**: totals, such as a conformer's E, the best energy in the molecule list, the QM energy, and H and G.
  - **Relative energies**: ΔE of conformers, the energy window, ZPE, H − E, G − E, and the electrostatic
    potential on surfaces. Entropy follows it: J/(mol·K) with kJ/mol.
  - Units: kJ/mol, kcal/mol, eV or hartree.
  - **Significant digits**: Automatic keeps the precision QuantaBench showed before, converted (0.01 kcal/mol
    becomes 0.01 kJ/mol). A number sets that many significant digits.
  - **Number style**: Automatic, Decimal or Scientific.
- Where it applies: the conformer table (ΔE and E headers and values), the molecule list's E column, the
  molecule's info line and the Properties page's thermochemistry, the energy window on the Conformer search page
  and in Settings > Build defaults, and the Surfaces panel's colour range, legend and ESP descriptors (σ² in
  (kJ/mol)²). Orbital energies (the HOMO-LUMO gap) stay in eV. The Properties page's list of descriptor
  columns keeps the column names, whose units are in the name (`esp_min_kcal`).
- **Display only.** `settings.display` is not a science setting, and grid workers never see it. CSV, JSON and SDF
  files keep their columns and units (`rel_kcal`, `best_energy_eV`, `esp_min_kcal`, `zpe_kcal`).
  - The energy window is still stored in kcal/mol (`energy_window_kcal`). Its box shows the display unit and gives
    back exactly the stored value until it is edited, so 6 kcal/mol shown as 25.10 kJ/mol is still 6.0, and a
    resumed run never sees a changed setting.
  - The surface colour range works the same way: the colours do not change with the unit.
- New `quantabench.units`: conversion factors (the computations' own: 23.060548 kcal/mol per eV,
  27.211386245988 eV per hartree, 4.184 kJ per kcal) and formatting. New `gui.energy_widgets.EnergySpinBox`.

### Update: package upgrades, and fork packages that arrive with a pull (2026-11-06)
- **`quantabench-cli update --upgrade-packages`** (Help > Check for updates: "Also upgrade the installed
  packages", or "Upgrade packages..." when there is nothing to pull) also upgrades what is installed. Until
  now an update reinstalled only when `pyproject.toml` changed, and pip left every package that still fit at
  its version.
  - It runs `pip install -e ".[<installed extras>]" <fork packages> --upgrade --upgrade-strategy eager`
    (`--upgrade-strategy only-if-needed` for the narrow one), with PyTorch pinned to the installed build.
  - The result lists what pip changed (`pip list` before and after). The app shows it and asks before it
    restarts.
  - Windows: if a file in use blocks pip, the message says to close the app and rerun from a terminal.
  - macOS: tblite's `libomp.dylib` is linked to PyTorch's again afterwards, as the installer does.
- **`scripts/fork-packages.json`** holds the packages the installers add on top of `pyproject.toml`, per
  platform. On Windows these are the dxtb fork (`qb-v0.4.0.post1`), tad-libcint 0.4 and tblite from the
  fork's `qb-v0.8.0-win1` release. On macOS, the dxtb fork, tad-libcint 0.4 and `--only-binary tblite`. On
  Linux, `tad-libcint<0.4`.
  - `install_windows.ps1`, `install_macos.sh` and `install_linux.sh` read the file instead of their own
    copies. Their pip lines and install stamps are unchanged, so nothing reinstalls because of this.
  - `update` reinstalls when this platform's entry changes in a pull. A package from a git tag or a release
    page whose entry changed is then reinstalled with `--force-reinstall --no-deps`: a new build keeps its
    version number (the coming OpenBLAS tblite, `qb-v0.8.0-win2`, PERF-2).
  - Every reinstall and upgrade keeps the fork packages on pip's command line, so a dependency change or an
    upgrade never swaps the dxtb fork for PyPI's dxtb. Without a readable file, `--upgrade-packages` stops.
- **Workers** keep running the code they started with; docs/WORKERS.md now says to restart them after an
  update. Options for updating them automatically are in BACKLOG UPD-1.

### Spartan's Essential 76 PFAS and the QM stage (VAL-2, 2026-11-06)
- **QuantaBench's Psi4 stage reproduces Spartan's ωB97X-D/6-31G*.** At Spartan's own geometries (TFA, PFPrA,
  triflic acid, PFHpA) the total energies agree to 5e-5 Eh with exact integrals and to 4e-4 Eh with the default
  density fitting. The dipoles agree to 0.001 D, with the same vector. Spartan's 6-31G* uses Cartesian d
  functions, like Psi4's.
- **Spartan import:**
  - The documents a multi-step conformer search writes its conformers to label them `Molecule1`,
    `Molecule41`, ... These labels are now resolved by structure like `M0003`. The Essential 76 set now gives
    76 wide rows, not 847.
  - Est. Density Functional single points (`SOPT DLXD2XV`) give their ωB97X-D/6-31G* SCF dipole (`QM_DIPOLE`)
    as `dipole_debye`, with the SCF's level in `qm_dipole_level`.
  - A level's best row is the one with the lowest `POSTENERGY`, the energy Spartan ranks conformers by.
  - `validate` pairs `qm_dipole_debye` with that level.
- **QM stage robustness:**
  - A failed free-atom SCF (MBIS volume ratios) no longer leaves its scratch files to the next SCF. One left
    the polarizability failing and Psi4 hanging at exit.
  - `run_driver` stops a driver that is still running 60 s after writing its result, and uses the result.

### Areas and volumes: a SASA bug fixed, Spartan's radii as an option (2026-11-06)
- **`sasa_A2` was wrong for every non-protein molecule.** It passed the radii of FreeSASA's ProtOr classifier,
  and those are 0 for any atom outside a protein residue, so the value was the union of bare 1.4 Å probe
  spheres on the atom centres. It is now the solvent-accessible area: Bondi radii plus a 1.4 Å probe. Columns
  from older runs keep the old value.
- **New columns:**
  - `vdw_area_A2` and `vdw_volume_A3`: the union of van der Waals spheres, with the run's radii set (Bondi by
    default).
  - `cpk_area_A2` and `cpk_volume_A3`: the same, with Spartan's radii table. Spartan's CPKVOLUME is reproduced
    to 0.1 % and CPKAREA to 0.5 % on the C3A B3LYP documents, at about 5 ms per conformer. `validate` compares
    them directly with Spartan's CPK values.

  Spartan's carbon radius is 1.92 Å against Bondi's 1.70, which is why its volumes ran about 7 % above
  QuantaBench's.
- **Settings `surfaces`:** the radii set (`bondi` or `spartan`), per-element overrides as in Spartan's
  editable table, and the SASA probe radius. Spartan's "accessible area" uses 1.0 Å. These reach
  `BuildConfig.radii_set`, `radii_overrides` and `sasa_probe_A`.
  - In the GUI: **Settings > Surfaces & radii**. It has the radii set, the SASA probe, and a 111-element table:
    Bondi, Spartan, and your value, with a search box and Reset selected / Reset all.
  - They are science settings. A resumed run keeps the values it started with, and a grid sends them to every
    worker, so all machines compute the same columns.
- **Version 0.3.0.dev3.** A run started before it is not resumed without `--force`, whatever its method,
  because its `sasa_A2` column would mix the old value with the corrected one.
- **ESP surfaces** gain `esp_polar_area_75_A2`, `esp_polar_area_100_A2` and `esp_polar_area_125_A2`: the area
  where |V| exceeds 75, 100 or 125 kJ/mol, as Spartan's polar areas.

### Dipoles from the validation run (300 molecules, gas and water, 2026-11-05)
- **Spartan ion dipoles are now about the centre of mass.** Spartan writes an ion's `DIPOLE_VEC` about the
  origin of the document's coordinates (its GUI hides the value, since it depends on where the molecule sits).
  The import now moves it to the centre of mass, QuantaBench's convention for ions, and keeps Spartan's own value
  in `spartan_dipole_origin_debye`. On the C3A B3LYP/C-PCM anions, 30 % moved by more than 1 D and some by up
  to 10.5 D. `validate` states the convention and warns when a wide table was imported before this change.
- **tblite conformers carry the real GFN2 dipole.** Each refined conformer used to store a Mulliken
  point-charge dipole; it now stores tblite's electronic dipole, moved to the centre of mass. The calculator
  already computes it, so this costs nothing. `dipole_debye_best` now equals the properties step's value, and
  `dipole_debye_boltz` averages real dipoles.
- What the validation showed for water (58 molecules, all ions):
  - At Spartan's own geometries, GFN2/ALPB reproduces B3LYP/C-PCM dipoles to within 0.41 D for anions
    (r 0.997) and 1.9 D for cations.
  - The remaining 4.2 D gap between QuantaBench and Spartan comes from the conformer each program picks.
    An ion's dipole about its centre of mass grows with the distance between the charge and the centre of mass.
- **Windows: xtb dipoles and gaps were silently charge-only.** xtb 6.7.1 writes its "program call" with raw
  backslashes, so `xtbout.json` was not valid JSON on Windows. Water came out at 1.6 D instead of 2.29 D. The
  file is now read anyway.
- **PubChem: `component_of_mixture`.** A CAS RN whose PubChem record has several components (a mixture, or a
  salt with an organic counter-ion) no longer counts as `different_structure` when QuantaBench's structure is
  one of the components. The note names the other components and flags when QuantaBench kept a smaller one than
  the largest. The 50-molecule trial found one such case: a dipropylammonium salt of HFPO-DA kept as the amine.
  The empty-cache line of `pubchem enrich` no longer says "cache: none".
- **Windows: short per-molecule time limits now trip.** They are timed with `perf_counter`, because
  `time.monotonic` ticks only every ~16 ms on Python 3.12.

### PubChem: identifiers, a structure check by CAS RN, and experimental values (surface tension groundwork, 2026-11-05)
- **`quantabench-cli pubchem enrich LIST --out enriched.csv`** looks every molecule of a CSV/Excel table, SD
  file or run folder up in PubChem. It finds the CID by CAS RN (cross-checked with the CIDs' registry
  numbers), InChIKey, SMILES or name, and writes `pubchem_*` columns:
  - the CID(s), title and CAS RNs (check digit verified);
  - formula, weights, SMILES and connectivity SMILES (PubChem's 2025 names; the old ones are still read),
    InChI(Key);
  - XLogP, TPSA, complexity, H-bond counts and charge.
- **`--validate`** says whether QuantaBench's structure is PubChem's for that CAS RN: `match`,
  `stereo_differs`, `charge_or_protonation_differs`, `different_structure`, `cas_not_in_pubchem` or
  `ambiguous_cas`.
- **`--experimental`** adds PubChem's experimental values (PUG View). Surface tension, density, mp, bp, vapor
  pressure, viscosity, solubility, logP, pKa and more each become a record:
  - the raw text, the parsed value and unit, and the value in mN/m (or g/cm3, °C, Pa, ...);
  - the temperature, the source and the reference;
  - for liquid properties, a condition class (`pure_liquid`, `aqueous_solution`, `unknown`) from the text, the
    reference and the melting point.

  `--records` writes them in long form. PFOA's "15.2 dynes/cm" (HSDB, from a review of fluorinated
  surfactants; PFOA melts at 54 °C) is flagged as probably an aqueous value at the CMC, not the pure liquid's.
- Within PubChem's usage policy:
  - at most 5 requests/s and 400/min, slower when its `X-Throttling-Control` header turns Yellow or Red;
  - 503/504 retried with backoff;
  - a `QuantaBench/<version>` User-Agent with the repository URL.

  Every answer is cached on disk (SQLite), so a 14,000-molecule pull resumes after an interruption and
  `--offline` reruns it without the network.
- `pubchem lookup ID` prints one compound. Name lookups in the GUI and CLI use the same rate-limited, cached
  client.
- docs/SURFACE_TENSION.md surveys sources of surface tension and lays out the modeling plan (QSAR-3):
  - which sources hold pure-liquid and which aqueous γ_CMC values;
  - the parachor and QuantaBench-descriptor baselines;
  - splits by cohort and class, and the applicability domain.

### macOS: one machine name (MacBook, 2026-11-05)
- On a Mac the hostname follows the network (on Tailscale it became `Brandons-MBP.tail....ts.net`), so the MacBook
  appeared in the registry as both `brandons-macbook-pro` and `brandons-mbp`. Machine names on macOS now come from
  the Bonjour name in System Settings (`scutil --get LocalHostName`).

### Spartan import: no more hangs in bond perception (full re-import, 2026-11-05)
- The re-import of all 1,067 documents stalled on one molecule: a tetrasodium dye whose four Na+ Spartan drew at one
  point, where RDKit's bond-order search never finishes (and its Hueckel mode crashes). Overlapping atoms (< 0.5 Å)
  now give no perceived bonds (no structure key, with the reason), and molecules with more than three P/S/As/Se atoms
  or over 150 atoms get connectivity only instead of the search. The stalled document now reads in 1.5 s.

### macOS: tblite and PyTorch share one OpenMP runtime (MacBook, 2026-11-05)
- tblite's macOS wheel and PyTorch each bundle `libomp.dylib`; a run that loaded both aborted with "OMP: Error #15"
  (the Mac's benchmark stopped at tblite-gfn2). `install_macos.sh` now links tblite's copy to PyTorch's on every run,
  and where both copies are still present tblite reports itself unavailable with the fix instead of aborting.
  Energies are unchanged (GFN2-xTB water identical to 1e-12 Eh with either runtime).

### Gas and water phases, paired per molecule (PFAS QSAR run, 2026-11-05)
- **`build --phases gas,water`** (any list of solvents; `gas` = none) runs the same input once per phase into
  sibling run folders, `RUN/gas` and `RUN/water`: normal resumable runs with their own job tables, so
  `--resume`, `--retry-failed`, lanes, chunks and `grid build --solvent` work per phase. Then it writes
  `RUN/paired.csv`. `build --phases ... --resume RUN` continues every phase and starts a missing one from a
  sibling's inputs and settings.
- **`quantabench-cli pair RUN/gas RUN/water [...] --out paired.csv`** rebuilds the paired table from finished or
  unfinished runs (read from their job tables), joining by molecule id and input InChIKey, never by row order:
  the identity columns once, `pair_status`/`pair_notes`, `dG_solv_kcal` (E_water − E_gas, each phase's best
  conformer at its own geometry), `dG_solv_G_kcal` (from G when both ran frequencies), `dG_solv_vertical_kcal`
  (fixed geometry), then `gas_<col>`, `water_<col>`, `delta_<col>`. The definition (ALPB's energy already holds
  the solvation free energy; 1 mol/L → 1 mol/L state) is in docs/ENGINES.md, "Phases".
- **The water phase is solvated in every step that can be.** Before, a `tblite-gfn2` run in water refined,
  computed its dipole/HOMO/LUMO and its frequencies in ALPB, but its polarizability was computed in the gas
  phase on purpose, and the GFN1-xTB `xtb_*` properties (charges, dipole, polarizability) came from dxtb in the
  gas phase without a word. Now the engine's polarizability is the finite-field response in ALPB (verified
  against −d²E/dF² to 1e-4 Å³), and in a solvent the `xtb_*` properties come from tblite's GFN1-xTB with the
  same solvation (dxtb and tblite agree to 5e-5 e in the gas phase). New: `tblite_gfn2_gsolv_kcal`,
  `tblite_gfn2_solvation`.
- summary.csv records `phase`, `solvent`, `solvation_model` and **`phase_notes`**: per step where it ran
  (`conformers: gas (MMFF94s has no implicit solvent ...); refine: ALPB(water); properties: ALPB(water);
  xtb_properties: ALPB(water), tblite GFN1-xTB; frequencies: ALPB(water)`) or why a step fell back to the gas
  phase.
- **Structure keys**: every summary.csv and descriptors.csv row has `inchikey` and `canonical_smiles` of the
  computed species and `input_inchikey` of the record as drawn, for joining to other tables (Spartan) by
  structure. Empty, with a note, when InChI cannot represent the species.
- **Metal counter-ions**: bondless Al³⁺, Zn²⁺, Fe²⁺/³⁺ and Cu²⁺ cations next to an organic ion are stripped like
  Na⁺/K⁺/Ca²⁺/Mg²⁺/Ba²⁺/Sr²⁺ and listed in `removed_fragments` (fosetyl-Al: one ethyl phosphonate anion). A metal
  next to neutral ligands (CO, phosphines, a corrin) or carbanions (Cp⁻, CN⁻) still fails as a metal complex;
  covalently drawn organometallics (Sn, Hg) stay whole.
- **pH 7 rules**: two basic sites within three bonds get one proton (piperazine, 1,2-diamines and guanidine were
  dications); 1,3-diamines keep two.
- **`--id-col CAS_RN` on SD files**: a record without the property takes its title as the id (it took DTXSID/ID
  first), with a note in the molecule's notes.
- Version 0.3.0.dev2. A solvent run (other than MMFF) started with an earlier version is not resumed silently: its
  gas-phase `xtb_*` and tblite polarizability columns would mix with solvated ones (`--force` continues anyway).

### Validation against Spartan: structure keys, `sample`, `validate` (VAL-1, 2026-11-05)
- **The Spartan tables carry structure keys QuantaBench can match**: `inchikey` and `canonical_smiles` of the
  structure as Spartan drew it (its bonds and bond orders, formal charges from each atom's bond-order sum, R/S
  and E/Z from the geometry), made by the same function as QuantaBench's species (`chem.validate.
  species_keys`). Ions no longer depend on rdDetermineBonds: a carboxylate, sulfonate or quaternary ammonium
  is keyed as drawn, and `to_mol` uses Spartan's bonds for them too.
- **`charge_check`**: `mismatch` when Spartan calculated another charge than the structure it drew
  (`structure_charge`); in the staged C3A sample, 12 of 47 ions (cations calculated as -1, phosphate dianions
  as -1). Those rows join but give no values in a validation.
- **Ambiguous structures get a wide row**: one per structure, `label_resolved` empty, `label_source` =
  ambiguous, `cas_candidates` = the CAS numbers it could be. `import` prints how many.
- `spartan_solvent` is QuantaBench's solvent name (`WASSER` -> `water`); `spartan_solvent_raw` and the level
  text keep Spartan's keyword.
- **`quantabench-cli sample`**: a reproducible validation set from the wide table, stratified by cohort (the
  document prefix, `C1A_`, ...) and charge class, with filters for Spartan values (PM6 / DFT-water), elements
  (AIMNet2's by default), charge classes and size; DFT-water molecules first within a stratum; writes SMILES +
  id + cohort for `build --file` and a design JSON.
- **`quantabench-cli validate`**: compares run folders, merged tables or the paired gas/water table with the
  wide table: joins by InChIKey, canonical SMILES and the stereo-free InChIKey with a CAS cross-check; gas vs
  PM6, ALPB water vs DFT in C-PCM water, solvation shifts (flagged: mixed levels); dipole, HOMO/LUMO/gap,
  polarizability, area/volume and charge extrema (correlation only where the quantities differ); N, MAE,
  RMSE, mean signed error, Pearson r, Spearman ρ, a linear fit and outliers, overall and per cohort and
  charge class. Writes `validation.csv`, `join.csv`, `summary.csv`, `report.md` and parity plots (new optional
  extra `quantabench[validate]` = matplotlib). See docs/VALIDATION.md.

### Runtime guards for the 14,223-molecule split (QSAR-2, SCALE-1 follow-ups, 2026-11-05)
- **Memory guard for the frequencies stage.** dxtb's analytic Hessian is the largest memory consumer of a
  molecule (measured: atoms x nao², 0.26-1.2 GB for 5-14 atoms, more than 6 GB for the PFOA anion). Over the
  lane's budget it now becomes central finite differences of the analytic forces (the memory of one energy +
  forces, about the same time on one thread); finite differences that do not fit skip the frequencies, and dxtb
  dipole derivatives that do not fit skip the IR intensities, each with a note. Measurements and the rule are
  in docs/ENGINES.md and `chem/memory_guard.py` (`frequencies_plan`).
- **Time limit for the conformer embedding.** RDKit ETKDG has no checkpoint, so a 90-heavy-atom peptide or a
  metal complex could embed for minutes past the per-molecule limit in one process. The embedding now gets a
  quarter of `resources.molecule_timeout_s` (450 s of the default 1800 s): one conformer is timed first, fewer
  are asked for when they would not fit, and RDKit's `EmbedParameters.timeout` is the hard stop. A note says
  when the embedding ran out of time and with how many conformers.
- **`lanes auto` prefers the faster device.** A GPU lane is planned only where this machine's benchmark says
  the GPU is the faster lane (`<method>@<gpu>` beats `<method>@cpu`): the MacBook gets CPU lanes only for
  AIMNet2 (6,387/h CPU vs 2,978/h MPS), NOVA keeps its GPU lane. The reason gives the numbers, or says there is no
  benchmark (then the GPU lane as before).
- **Benchmarks in water, and a mixed benchmark set.** `machines --bench M --solvent water` records
  `M+water@<device>`, never mixed with gas-phase keys; methods without a water model (and runs that fell back to
  gas phase) give no water number. `--bench-set mixed` adds neutral baseline organics and cholesterol to the
  five PFAS anions; the set and the solvent are recorded with each number. Speed-aware lanes, `lanes auto`,
  `grid build --weights auto` and `build --chunk-weights auto` read the keys of the run's solvent.
- **Settings > Compute** has "Memory per lane" (`resources.memory_budget_gb`; 0 = auto, Off = no guard) and
  "Time limit per molecule" (`resources.molecule_timeout_s`; Off = none).

### Apple GPU: AIMNet2 on MPS (first run on the MacBook, 2026-11-05)
- AIMNet2's Warp-free path failed on the Apple GPU with "Undefined type Double": `t.to("cpu", torch.float64)` casts on
  the device before the copy, and MPS has no float64. Tensors now move to the CPU first and are cast there (and the
  batched optimizer casts on the CPU side when bringing energies to the GPU). A test keeps the pattern out.
- The benchmark no longer "times" dxtb on a GPU where QuantaBench keeps dxtb on the CPU (MPS, AMD's Windows wheel):
  it records why instead of a fell-back-to-CPU error.

### Spartan QSAR documents: macOS streams, levels of theory, CAS join, wide table (IMPORT-2, 2026-11-05)
- **Documents saved by Spartan for macOS are read.** They hold `Input`, `PropertyArchive`, ... uncompressed
  where Spartan for Windows writes `Input.gz`, ...; all of those molecules (the PM6 and conformer documents
  of the PFAS QSAR project among them) were skipped as "no Input.gz". A molecule that still cannot be read is
  counted under a named reason, which the CLI prints.
- Every row carries its **level of theory** (`spartan_version`, `spartan_task`, `spartan_method`,
  `spartan_basis`, `spartan_solvent`) and `spartan_status` (completed, or failed with Spartan's message).
- **QSAR properties of PM6/DFT jobs** take QuantaBench columns where quantity and unit match: `homo_ev`,
  `lumo_ev`, `gap_ev` (from hartree), `dipole_debye`, `zpe_kcal`, `h_corr_kcal`, `g_corr_kcal`,
  `entropy_cal_molK`. The rest (CPK area/volume, ESP areas, Spartan's estimated polarizability, ...) stays
  `spartan_*` with its unit documented.
- **Labels are resolved**: rows labelled `M0003`/`M0001[2]` by Spartan are joined by structure to the CAS
  label of their base document (`C1B_P5_*.spartan` -> `C1B_P5.spartan`): `label_resolved`, `label_source`
  (input, structure, ambiguous, unresolved), `label_note`, `structure_key`. Ambiguous structures are never
  guessed.
- **`import --wide qsar.csv`**: one row per label, each level's best result as prefixed columns (`conf_*`,
  `pm6_*`, `dft_*`, ...), prefixes from the method and task.
- Faster on large documents: stream lookups no longer scan the document's directory each time (olefile's
  lookup is linear in the molecules of a document).

### Batches survive molecules that blow up memory or run forever (QSAR-2, 2026-11-05)
- **Memory guard for dxtb** (`chem/memory_guard.py`). dxtb's full-mode autograd keeps every SCF iteration, so
  memory grows as iterations x nao². The GFN1 polarizability took 1.5 GB for deca-alanine and for the C20 PFCA
  anion, and more than 1.9 GB for a dication. Before each dxtb step the pipeline estimates its memory from the
  basis size, charge and spin (a rule fitted to measurements, see docs/ENGINES.md) and compares it with the
  budget of the lane: by default the available memory divided by the lanes, at least 1.5 GB, set with
  `resources.memory_budget_gb` or `build --memory-budget GB`, < 0 = off. Over the budget:
  - the xTB polarizability comes from **tblite**'s GFN1-xTB finite field, the same quantity (39.1012 vs
    39.1009 Å³ for Ala10) at ~1/40 of the memory, with a note and `xtb_polarizability_method`;
  - without tblite, the polarizability is skipped with a note (or the whole properties step, if even the
    charges do not fit); the molecule stays ok;
  - dxtb refinement: batched runs use smaller batches. A molecule where not even one conformer fits fails with
    `MemoryGuardError: ... estimated X GB ...`, and `auto` tries its next engine instead.
- **Per-molecule time limit:** `resources.molecule_timeout_s`, `build --molecule-timeout S`, default 1800 s,
  0 = off. A lane still on one molecule after the limit is killed and restarted. In one process the molecule
  stops at its next checkpoint (best effort). Either way the molecule is recorded failed with
  `time limit (1800 s) in <stage>`. This is not a crash: it is not retried in the run, and `--retry-failed`
  redoes it. The GUI batch gets both settings from settings.json.

### Fixed: the GUI reset the Pipeline settings at start-up (2026-11-05)
- Opening the window copied the Pipeline widgets into the settings before they had been filled from them (the
  Machines dock asks for the current configuration while it is built), so pH, protonation, conformer count,
  temperature, steps and seed took their minimums (pH 0, 1 conformer, 1 K), and closing the window saved them.
  CLI runs were not affected. If you closed the GUI since the Machines dock arrived, check Settings and the
  Pipeline values once.

### Windows: non-ASCII names (stress sample on NOVA, 2026-11-05)
- XYZ, extXYZ and LAMMPS files are written as UTF-8 on every OS: on Windows a molecule named "β-..." failed its
  export (ASE opened the file in cp1252).
- Molecule ids (folder and file names) spell Greek letters out (β-Tocopherol -> `beta-Tocopherol`), drop accents
  and never start with "-".

### Research inputs: salts, metals, charges, spin (stress sample, 2026-11-05)
From a 325-molecule stratified sample of the PFAS QSAR set (PFAS and baseline cohorts: salts,
organometallics, ions from -5 to +2, up to 157 heavy atoms), run through MMFF, GFN2-xTB (tblite), GFN1-xTB
(dxtb), AIMNet2 and `auto`:
- **No metal is dropped silently.** Keeping the largest fragment of a complex drawn as ions computed CO for
  metal carbonyls, a bare corrin for cobalamins and cyclopentadiene for ferrocene; those records now fail
  with `metal complex: ... would remove Co (...)` and the batch goes on. Alkali/alkaline-earth ions are
  counter-ions as before.
- **Organometallics are computed whole.** Only bonds to group 1/2 metals are disconnected; RDKit's default
  also split Hg-O and Sn-O, so R-Hg-OAc became R-Hg+ (charge +1) and bis(trialkyltin) oxides half a molecule.
- **What standardization did is in the notes and `summary.csv`:** new columns `input_charge` (as drawn),
  `multiplicity`, `removed_fragments` and `forcefield`; notes such as `removed 2 x [Br-]` and
  `charge +0 as drawn, -1 computed (counter-ions removed, then neutralized and protonated for pH 7)`.
- **Spin**: drawn radicals give the high-spin multiplicity (it was always 1), an odd electron count a
  doublet, with a note; an impossible explicit multiplicity fails the molecule.
- **`auto` says why** it passed an engine over ("AIMNet2 ... has no parameters for Sn; using GFN2-xTB").
- **MMFF94s -> UFF** is noted (the energies of method `mmff` are then UFF's; descriptor `method` = UFF). With
  no force field at all (e.g. Cr(III) complexes) method `mmff` fails instead of reporting 0 eV.
- **Cyclopentanes embed again**: RDKit 2026.09's ETKDG small-ring torsions abort plain cyclopentanes
  (cyclopentane, cyclopentanol, cis-1,2-cyclopentanediol) with "Invariant Violation: bad direction in
  linearSearch" for every seed; the embedding is repeated without small-ring torsions, with a note.
- R groups, `*` and molfile atom lists fail with "not a definite structure" (ETKDG aborted with an RDKit
  invariant violation and a C++ stack trace in the log).
- GFN1-xTB polarizabilities that cannot be physical (4x10^5 Å³ for a gas-phase phosphate dianion) are
  discarded with a note.
- **SD files**: unreadable records are reported (record number, title, RDKit's reason) instead of vanishing;
  `--id-col` works for SD properties (e.g. `--id-col CAS_RN`); repeated titles get `_2`; RDKit's
  per-record hydrogen warnings and UFFTYPER messages no longer flood the log.
- An AIMNet2 model that cannot be loaded (weights not downloadable offline) fails the next molecules at once
  with the reason and "--retry-failed", instead of retrying the download for every molecule of a batch.
- The "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver" line every aimnet lane printed on
  AMD/CPU machines is filtered (other Warp output passes through).

### The GUI at 14,000 molecules (SCALE-1, 2026-11-05)
- **The Molecules dock is a virtualized table** (`gui.molecule_list`): one light row per molecule (input
  order, status, id, conformers, best energy, name, method, formula) instead of a list item plus the full
  result in memory. Sort by any column; search id, name, SMILES or formula; filter built / failed / not
  built. Only the selected molecule and the last 12 viewed keep their full result; the others are read from
  their folder when selected, off the GUI thread (a few ms). A batch's finished molecules update their row
  instead of keeping their result (a 400-molecule MMFF batch: +8 MB; the rest is a backlog follow-up).
- **File > Open run folder...** lists a run folder, finished or partial, from its job table (or the
  summary.csv of a plain batch). A synthetic 14,000-molecule folder opens in 0.6 s (first molecule shown
  at 1.0 s) and adds 31 MB to the window's memory.
- **Build > Resume run folder** continues the opened (or last) run folder's unfinished molecules with its
  stored science settings: the 0.2 item "resume a run folder from the GUI".
- New read helpers: `export.read_molecule` (a molecule's folder back into a `MoleculeResult`, every
  conformer from the SDF), `jobs.molecule_rows`, `jobs.load_result`, `jobs.load_inputs`, `jobs.stored_config`.

### GUI start-up memory and selection speed (SCALE-1 follow-ups, 2026-11-05)
- **The window no longer imports torch, dxtb or tad-libcint** (~500 MB) to label engines and GPUs. Menus and
  labels use `engines.available(key, quick=True)` / `libcint_status(check_import=False)`: installed versions
  and files only (`libcint_status_from_versions`). The GPU list and the full libcint check run in a child
  Python process after the window shows (`resources.detect_isolated`, killed when the window closes); its
  verdict replaces the version-based guess, so an unusable GFN2 is still marked, with its reason. Runs and
  `quantabench-cli doctor` keep the full checks. Idle memory with a 14,000-molecule run folder open: 1,133 →
  634 MB (Linux, software OpenGL), of which VTK is ~355 MB; without VTK the window would be ~280 MB.
- **One 3D render per selection** (was 3-9): render requests within one event-loop pass share one render, the
  camera fit no longer renders by itself, and pyvistaqt's default 5-per-second idle re-render is off.
  Selecting a molecule: median 557 → 224 ms under software OpenGL; scrolling the 14,000 rows 10 → 2.5 s.
- The Log view keeps 5,000 lines (was 20,000): each line holds ~7 KB, so the old cap let a long batch's log
  grow to ~140 MB; this was most of a GUI batch's per-molecule memory growth (~3 lines, ~21 KB per molecule).

### Spartan documents in one table (IMPORT-2, 2026-11-05)
- `quantabench-cli import *.spartan --table out.csv [--sdf out.sdf] [--conformers all]` reads any number of
  Spartan '24 documents into one CSV: one row per molecule and document, keyed by the label, with
  `source_file`, `molecule_index` and a `duplicate_label` flag (all rows kept). Scalar Spartan properties
  become `spartan_<name>` columns; conformer libraries report the conformer count and the lowest-energy
  conformer. Streams molecule by molecule (~0.7 s and < 100 MB for a 100-molecule batch).
- File > Import calculation output accepts `.spartan` documents (one result per molecule, conformers ranked).
- New core dependency `olefile` (pure Python, BSD) for the OLE container.

### Speed-aware lanes (DIST-5, 2026-11-05)
- With lanes of different speed (`--lanes auto` for AIMNet2 on a GPU machine: one GPU lane plus CPU lanes), a
  slower lane no longer starts one of the last molecules when the faster lanes would finish everything left
  before it: a small batch now ends no later than on the GPU lane alone. Speeds are measured per kind of lane
  as molecules finish (seconds per atom), seeded from this machine's benchmark in the registry
  (`machines --bench`); unknown speeds behave as before. A fast lane that dies or stalls releases the held
  molecules. One log line says when lanes start holding back; the run ends with the measured lane speeds.
- On by default: `build --no-speed-aware-lanes`, Settings > Compute "Speed-aware lanes" or
  `resources.speed_aware_lanes` turn it off. Lanes of one kind (`--lanes 6`) are not affected.

### Installing with the dxtb fork (2026-11-05)
- Fixed `pip` `ResolutionImpossible` when installing `.[all]` together with the dxtb fork (Windows and macOS
  installers): the `ml` extra no longer lists `tad-dftd3`, whose every release pins an old tad-mctc. AIMNet2's
  Warp-free path takes D3 from the fork's vendored tad-dftd3 (else PyPI's), and the ROCm/MPS linear-algebra
  routing also patches the fork's vendored D3 weights, `storch.eighb` and EEQ guess.
- The Windows and macOS installers remove a tad-dftd3/tad-dftd4 left over from PyPI's dxtb once the fork is in.

### Large batches and joining machines (2026-11-05)
- A worker measures its machine's throughput on its first start (a machine joining the pool), holding jobs
  meanwhile; `worker --bench auto|always|never`. Machines dock: "Benchmark this computer"; this computer is
  always listed, "not measured" until it is.
- Batch runs from the CLI, the GUI and workers no longer keep every molecule's RDKit ensemble and normal
  modes in memory (`run_resumable(keep_results=False)`, `jobs.light_result`): everything is on disk already.
- GUI: Python's cyclic garbage collection runs in the GUI thread on a timer, so a closed window or panel is
  never destroyed by a worker thread (Qt aborts the process when that happens; seen in the test suite).

### Review fixes for lanes, weighted splits and GPU paths (0.3.0.dev1, 2026-11-05)
- **Named `--chunk-weights` are a mapping**: `a=3,b=1` and `b=1,a=3` give the same split (chunks are the
  names sorted; this machine's is found by name). In typed order, two machines could overlap and leave a
  quarter of a list out.
- **`--chunk-weights auto` is resolved once per list**: the first machine stores the split in
  `<settings>/machines/splits/`, the others reuse it even after a re-benchmark (SyncThing conflict copies:
  the oldest wins, with a warning).
- Equal weights: the printed plan, `grid build`'s split and the worker's weights filter now select exactly
  what runs (the modulo split).
- **Lanes**: an error raised by a callback (a closed stdout, a full disk under the job log) no longer counts
  as a lane crash (it marked whole batches "crashed the process 2 times"); a molecule sent to a lane that
  had just died goes back to the queue uncounted; external programs (xtb, CREST, Psi4) a dead or killed lane
  left running are killed, `wsl:` programs on the Linux side too; at most 31 lanes on Windows.
- `--lanes auto` never mixes a float32 dxtb GPU lane with float64 CPU lanes (CPU lanes, or one GPU lane
  with dxtb GPU = on).
- The aimnet neighbor-list guard is applied to the class, so lists aimnet creates later are covered too.
  Resuming an `aimnet2`/`auto` run started before 0.3.0.dev1 is refused unless forced: its earlier AIMNet2
  energies may carry the neighbor-list error.
- Settings > Compute's dxtb linear-algebra and GPU modes now apply to every CLI command and GUI action, not
  only to builds.

### Splitting work across cores, GPUs and machines; forks for Windows and macOS (2026-11-05)
- **Lanes** (`dist.lanes`, `build --lanes auto | N | gpu:1x4,cpu:5x2`, Settings > Compute > Parallel
  lanes): a batch runs in parallel worker processes that share this machine's threads and GPU; the parent
  stays the only `jobs.sqlite` writer, a crashed lane is restarted. NOVA: MMFF 3.7x, GFN1-xTB 3.45x.
- **Machine registry and weighted chunks** (`dist.profile`, `dist.plan`): `quantabench-cli machines
  [--refresh] [--bench]` writes each machine's hardware, engines and measured molecules/hour to
  `<settings>/machines/`; `build --chunk 1/2 --chunk-weights auto` (or `3,1`, `btm-nova=3,...`) splits a
  list in proportion. The weights are stored with the run.
- **Workers over Tailscale** (`dist.worker`, `dist.client`): `quantabench-cli worker` (token, Tailscale
  address only) and `grid build|status|cancel|merge`, which splits a list across machines, follows them and
  merges the tables (synced run folders are read in place, others downloaded). docs/WORKERS.md.
- **Machines dock** in the GUI: the registry, worker status and this computer's lane plan.
- **AIMNet2 on AMD and Apple GPUs** without NVIDIA Warp (`chem/aimnet_torch.py`): ~10x the CPU on an
  RX 9060 XT. **Fixes a stock aimnet neighbor-list bug** that gave wrong energies for larger molecules
  evaluated after smaller ones in the same process (CPU): re-run earlier multi-molecule AIMNet2 CPU batches.
- **dxtb on ROCm/MPS** through CPU routing of the missing linear algebra (`chem/torch_compat.py`); off by
  default because dxtb 0.4 is slower on those GPUs than on the CPU (Settings > Compute > dxtb on the GPU).
- **GFN2-xTB, polarizability and IR intensities in dxtb on Windows and macOS** with the dxtb fork
  (`brandonbtm29/dxtb@qb-v0.4.0.post1` + `tad-libcint>=0.4`); `doctor` says what is missing.
- **tblite on Windows** from the fork's wheels (`brandonbtm29/tblite` release `qb-v0.8.0-win1`).
- **One-command installers** `scripts/install_macos.sh`, `install_linux.sh`, `install_windows.ps1`
  (`--with-engines`, `-Gpu rocm` for a second `.venv-rocm`), launchers without a terminal window,
  **Help > Check for updates** / `quantabench-cli update`.
- CI: `external-engines.yml` (xtb, CREST, MOPAC, Psi4 on Ubuntu and macOS).
- `doctor` shows which AIMNet2 and dxtb path each GPU gets.
- Resuming a run now also compares solvent, CREST and QM settings (they change the science).
- Windows + ROCm: QuantaBench's entry points exit with `os._exit` after using the GPU, because AMD's runtime can hang at process exit (`QUANTABENCH_HARD_EXIT`). The console/GUI scripts now point at `cli:entry` / `gui.app:entry` (`pip install -e .` once after updating).

### External engines, QM stage, surfaces and spectra (2026-11-01)
- **External-engine layer** (`chem/external.py`): finds xtb, CREST, MOPAC and Psi4 (settings,
  `QUANTABENCH_<NAME>`, PATH, conda/mamba envs, `wsl:` on Windows), runs them with thread caps,
  `OMP_STACKSIZE`, scratch folders, log files, timeouts and process-tree cancellation. Settings > Engines
  tab with "Detect now"; `doctor` lists the programs.
- **Engine plugins** (`engines.register`): `xtb-gfn2`, `xtb-gfn1`, `gfnff` (xtb program), `tblite-gfn2`,
  `tblite-gfn1`, `mopac-pm7`, `mopac-pm6`. Each brings its own properties (`gfn2_*`, `tblite_gfn2_*`,
  `pm7_*` descriptors) and, for xtb and MOPAC, its own optimizer (`--optimizer native`).
- **Implicit solvent** for every engine that has one (`--solvent`, GUI Refinement > Solvent): ALPB/GBSA,
  COSMO, PCM; gas phase with a note otherwise.
- **CREST** conformer search (`--conformers crest`, `--crest-method`, GUI Conformer search page) with
  RDKit fallback.
- **QM stage with Psi4** (`--qm wb97x-d/def2-svp`, `--qm-task`, `--qm-top`, `--qm-freq`, `--cubes`; GUI
  "QM (DFT)" page): energies or optimizations, dipole, charges (Mulliken/Löwdin/MBIS), orbitals,
  polarizability, PCM, frequencies + IR + thermochemistry, density/ESP cubes; `qm_*` descriptors and
  `<id>_qm_*` files.
- **Surfaces**: Gaussian cube reader/writer, 0.002 au density surfaces with ESP and GIPF descriptors
  (`qm_surf_*`), point-charge ESP fallback, Surfaces panel (red = negative, as in Spartan).
- **Spectra panel**: IR spectrum with sticks, QM trace, experimental overlay (CSV, JCAMP-DX incl.
  compressed forms), click a peak to animate its normal mode.
- **Import** of other programs' outputs through cclib: File > Import calculation output,
  `quantabench-cli import`.
- `envs/` conda files for the external programs; `docs/ENGINES.md`.
- `pyqtgraph` is a core dependency; `tblite` joins the `xtb` extra (not on Windows); new `interop` extra.

## 0.2.0 (unreleased)

### Fixed on native Windows (2026-10-27)
- Pin `pyvistaqt<0.13`: its new framebuffer OpenGL widget crashes on AMD/Windows; 0.12 works.
- xTB polarizability and IR intensities degrade with a note when dxtb's libcint backend is missing
  (native Windows), instead of failing; GFN1 Hessians no longer request dipole integrals.
- Dark-mode 2D viewer test no longer assumes a light desktop.
- LAMMPS SMD: charged single atoms (counter-ions) use Joung-Cheatham ion LJ parameters (`Na_ion`, ...)
  and stay on `nve/limit` in every stage. With UFF's neutral-atom LJ, Na+ collapsed onto a carboxylate
  oxygen in vacuum and the run blew up ("Out of range atoms") for 3 of 6 seeds on Windows.

Viewer:

- 2D structure view (RDKit SVG depiction) next to the 3D view, both sharing one atom selection: pick
  in either view, see the highlight in both. Grab-to-rotate in-plane, pan, zoom, CIP and charge notes,
  SVG/PNG export. Long perfluoroalkyl chains are drawn as clean 120° zig-zags.
- 3D viewer rewritten (`gui.viewer3d`): Avogadro/PyMOL mouse and keys, double-click to recenter,
  axis views, perspective/orthographic, exact NumPy picking, Rotate / Select (box) / Move (rigid
  rotate and slide of the selection or adsorbate) modes, distance/angle/dihedral readout in the status
  bar, labels (element, index, charge), element/charge/tag coloring, instanced rendering for 20k-atom
  slabs, 2x screenshots. `gui.viewer.MoleculeViewer` remains as a shim.
- Toolbar: mouse mode group (Ctrl+1/2/3), label and color modes, Show 2D (Ctrl+D).
- Surface builder is modeless and previews in the 3D view; "Move adsorbate" lets you orient the
  molecule by hand and Save uses the moved coordinates. New "Export LAMMPS SMD setup..." button.

Chemistry and runs:

- Batched conformer refinement (`chem.batch_opt`): vectorized L-BFGS and FIRE over all conformers of
  a molecule in one torch batch (dxtb, AIMNet2); identical results to sequential ASE LBFGS, 2.4x faster
  on CPU for PFOA x 8 conformers. `BuildConfig.batched` = auto (GPU only) / on / off; `--batched`.
- Frequencies and thermochemistry (`chem.vibrations`): exact dxtb Hessian with a memory guard, finite
  differences otherwise, IR intensities and spectra, RRHO/qRRHO ZPE, H, S, G. `build --freq`, new `freq`
  command, a Properties-page checkbox, `vib_*` descriptors, `<id>_ir.csv` and `<id>_modes.csv` (export
  format `spectrum`).
- Resumable batch runs (`chem.jobs`): SQLite job table, crash-safe commits, run lock, config guard,
  `--chunk k/n` across machines, `--resume`, `--retry-failed`, `status` and `merge` commands. GUI
  batches write the job table too.
- Excel input (`.xlsx`, `.xlsm`) and CompTox-aware tables: DTXSID and preferred name, SMILES →
  QSAR-ready → MS-ready fallback per row, duplicate ids handled, skipped rows reported.
- LAMMPS steered-MD bridge (`quantabench.md`): rigid adsorbate pulled off a frozen graphite slab, UFF
  and graphite LJ presets, PPPM slab, GPU package flags; log and pull-force parsing with detachment
  work and peak force. `md smd`, `md parse`, `slab --lammps-dir`.
- `doctor` lists lammps and openpyxl. New extras `md` and `excel` (both in `all`).

Fixes:

- `.gitignore` no longer hides `src/quantabench/build/` (the surface builder was missing from 0.1.0
  commits).

Review fixes (science, GUI, robustness):

- Frequencies no longer overwrite the best conformer: its ranking energy, charges and 3D descriptors
  stay paired with its exported geometry. The tightly re-optimized frequency geometry is kept in the
  summary (`vibrations.positions_A`), written as `<id>_freq.xyz`, and its shift is the
  `vib_geometry_max_shift_A` descriptor.
- Single-atom ions get H and G with their electronic energy, like every polyatomic molecule.
- The rotational symmetry number σ is detected per molecule from its point group (`symmetry_number`
  0 = auto, the new default in the CLI, GUI and `BuildConfig`); `vib_symmetry_number` records it.
- pH 7 protonation keeps halide anions (no more HCl/HBr from the Uncharger) and turns NH3 into NH4+.
- `md parse` / `md smd --run` subtract the thermostat's Langevin drag work from the detachment work
  (and print the raw W next to it); `smd_setup.json` stores the drag force.
- GUI: worker callbacks (adding a SMILES/name, opening files, hardware detection) run on the GUI
  thread (they ran on the worker thread, unsafe on Windows); Move mode grabs the whole molecule
  pressed on and a click no longer changes the selection, with a warning in the Surface Builder if a
  move changes a bond; Cancel is Shift+Esc (Esc clears the selection) and asks before stopping a batch;
  closing while work runs no longer freezes the window or aborts the process; closing the Surface
  Builder during the LAMMPS export no longer aborts; its preview survives selecting another molecule;
  the selection is cleared when the preview is replaced by the built molecule; 3D labels are no
  longer scaled twice on HiDPI screens; the 3D view follows light/dark theme changes.
- Runs: `status` opens `jobs.sqlite` read-only (it used to checkpoint and delete the -wal of a synced
  copy); a molecule that crashes the process twice is marked failed instead of blocking every resume;
  the run lock records Windows vs WSL2; `merge` reports missing or unfinished run folders (and reads
  their job tables) and exits non-zero.
- CLI: `--no-freq` and `--protonate/--no-protonate` override settings saved by the GUI; `--sheet 1`
  is a 0-based index; `freq --as-is` reports unreadable files and falls back from a missing default
  engine.
- Inputs: all-letter SMILES (CCCO, OCCO, NCCN, ...) are no longer sent to PubChem as names, and one bad
  line in a .smi/.txt list is reported and skipped instead of aborting the batch.
- Tests no longer overwrite the real GUI window layout (`QUANTABENCH_WINDOW_STATE`); CI installs Mesa
  OpenGL on Windows and has job and per-test timeouts.

## 0.1.0 (unreleased)

First public scaffold.

- Geometry builder pipeline:
  - inputs: SMILES, PubChem names, structure files, CSV lists
  - standardization and PFAS-aware protonation
  - ETKDGv3 + MMFF94s conformer search with energy-window and RMSD pruning
  - refinement with AIMNet2, GFN1/GFN2-xTB (dxtb) or MMFF, with GPU-to-CPU fallback
  - Boltzmann weights, descriptors, and GFN1-xTB charges, dipole and polarizability
  - export to XYZ, extXYZ, SDF, PDB, MOL, LAMMPS data, and batch CSV tables
- Surface builder: AB graphite and metal slabs, adsorbate placement, overlap report.
- PySide6 GUI with an OVITO-style layout, PyVista viewport, and optional embedded OVITO viewport.
- CLI: `doctor` (with `--bench`), `build`, `slab`, `settings`.
- Resource manager: thread caps and GPU device policy (ROCm, CUDA, MPS) with a memory cap.
- Settings can live in a synced folder through a pointer file.
