# Architecture

```
            +----------------------- GUI (PySide6) -------------------------+
            | Build     |  Molecules dock | 2D view | 3D view | workspace's  |
            | Calculate |                 | playback bar      | docks        |
            | Simulate  |  Conformers, Log; Analyze: Table, Plot, Spectra    |
            | Analyze   |  Settings | Surface builder | Library | Machines   |
            | Library   |  (activity bar: gui.workspaces)                    |
            +-----------------------------+---------------------------------+
                                          | QThread workers (never block the UI)
            +--------------- CLI (argparse) --------------+
                                          |
                     +--------------------v---------------------+
                     |  chem.jobs.run_resumable (batches, SQLite)|  crash-safe, chunkable
                     |  chem.pipeline.run(inputs, BuildConfig)   |  UI-agnostic, cancellable,
                     |  progress / log / on_result callbacks     |  per-molecule error capture
                     +--+------+------+-------+-------+------+---+
                        |      |      |       |       |      |
                  inputs  standardize conformers engines properties export
                                    crest   batch_opt  vibrations   qm -> surfaces/cube
                                    eng_xtb, eng_tblite, eng_mopac (plugins)
                        \_______ external: subprocess runner for xtb, CREST, MOPAC, Psi4 _______/
                     build.surfaces -> md.lammps (steered-MD setup) -> md.lammps_log
                                          |
                     +--------------------v---------------------+
                     |  resources: threads, GPU device, memory   |
                     |  settings: JSON in a (synced) folder       |
                     +-------------------------------------------+
```

## Packages

| Module | Responsibility |
|---|---|
| `quantabench.resources` | Hardware detection (CPU, RAM, WSL, PyTorch backend: ROCm/CUDA/MPS), thread caps for BLAS/OpenMP/PyTorch, device selection with memory cap, matmul benchmark. |
| `quantabench.settings` | `Settings` dataclasses, JSON load/save, and the settings-folder pointer (env var, then repo-local pointer, then per-user config). |
| `quantabench.priority` | Run priority (`normal`/`low`/`background`) and free threads, per machine: the thread budget and its cap for the auto plans and borrowed CREST threads, lowering a lane, CLI or worker process (`apply_to_process`) or the GUI's build thread (`thread_lowered`), and how `run_program` starts a program at that priority (`child_options`: Windows priority class and I/O priority, `nice`/`ionice` or `taskpolicy` in the POSIX and WSL wrappers). |
| `quantabench.dist` | Lanes, machine profiles, the weighted split, the HTTP worker and `grid build` (docs/WORKERS.md). `dist.control` (DIST-7): this machine's pause file, the status files, `ClockWatch` (sleep and wake), the battery, the split choices and learned rates, `machines set` and `worker pause|resume`; `jobs.BatchControl` pauses a running batch; `client.MachineWatch` moves a paused or away machine's molecules while a grid is followed. |
| `quantabench.sensors` | Live machine metrics (MON-1): `read()` (never raises, about a second at most) and `Monitor` (a sampling thread with 10 minutes of history) over per-OS providers: psutil, Windows performance counters (ctypes PDH) and LibreHardwareMonitor's web server, macmon on macOS, cpufreq/hwmon/RAPL/amdgpu sysfs and nvidia-smi on Linux. Used by `cli_monitor` (`monitor`, `doctor`), `gui.machine_monitor`, the worker's `/v1/sensors` and the run timings' load samples. |
| `quantabench.units` | Energy units and number format for the GUI (`Settings.display`): conversion factors (the computations' own), `EnergyDisplay` (unit per kind, text, headers). Display only: results and exports keep their units. |
| `chem.inputs` | `MoleculeInput` records from SMILES, InChI, InChIKeys and PubChem names, structure files, CSV and Excel (CompTox-aware: DTXSID, preferred name, SMILES-column fallback per row, `ReadReport` of skipped rows). `read_raw` reads a file as it is; `read_file(..., options=)` applies the import options. |
| `chem.formats`, `chem.import_options` | More structure formats (MOL2, Marvin, ChemDraw CDX/CDXML, CML, CIF, Gaussian/ORCA inputs; Open Babel for the rest) and the Spartan-style import options (2D to 3D, hydrogens, keep coordinates, data fields, filters, atom order, structures as conformers), with `survey` deciding whether File > Open asks. |
| `chem.interchange` | Structures as text and bytes, for the clipboard and drops: `parse_structure_text` (MOL V2000/V3000, SD, CDXML, Marvin MRV, else `inputs.parse_text`), `parse_cdx_bytes` (binary ChemDraw: `cdx_to_cdxml` translates it for RDKit's CDXML reader, since RDKit 2026.09's Python wrapper reads only CDXML), MDLCT, and `to_smiles` / `to_molblock` / `to_sdf` / `to_cdxml` (2D). |
| `chem.edit`, `chem.fragments` | The model kit's edit model (UI-7): `EditableMolecule`, an RDKit molecule with one 3D conformer changed only through undoable `Command`s; main-group atoms by VSEPR, metals as coordination centres (`COORDINATION`, UFF in Minimize) (attach / replace / delete, bond order, charge, make / break bond, set distance / angle / dihedral, positions, Minimize, local relax), with VSEPR free valences (`free_valences`) and `to_input()` for the pipeline; `History` is its undo stack outside the GUI. `chem.fragments`: the groups, rings, substituents and PFAS pieces as SMILES with `[*:1]` attachment points. GUI-free. |
| `chem.pubchem`, `chem.pubchem_enrich` | PubChem PUG REST + PUG View client within the usage policy (rate limit, `X-Throttling-Control` backoff, SQLite response cache, offline mode); CAS RN check digit; experimental records with units, temperature and condition class; `enrich` (CIDs, `pubchem_*` columns, structure check by CAS RN) behind `pubchem enrich`. |
| `chem.standardize` | RDKit cleanup, largest fragment, neutralize, and the pH protonation rule table (or Dimorphite-DL). |
| `chem.conformers` | ETKDGv3 embedding, MMFF94s/UFF, energy-window plus RMSD pruning, Boltzmann weights. |
| `chem.engines` | Engine registry and ASE calculator factory: `DxtbCalculator`, AIMNet2, and `register()` for plugin engines (`PLUGINS`), each with an availability check, a calculator factory, optional properties and native optimizer. `method_keys`, `auto_candidates` (solvent-aware). |
| `chem.external` | External programs: discovery (settings, `QUANTABENCH_<NAME>`, PATH, conda envs, `wsl:`), `program_env` (threads, `OMP_STACKSIZE`, `PSI_SCRATCH`, `WSLENV`), `work_dir`, `run_program` (log file, timeout, process-tree cancellation, the run priority through `popen_args`; `program_cwd` for a folder inside WSL). |
| `chem.wsl_scratch` | CREST-SCRATCH-1: a `wsl:` program's job folder inside WSL (`/dev/shm/quantabench` when it and WSL's memory have room, else WSL's ext4 disk; `engines.crest_wsl_scratch`), with the inputs copied in, only the results copied back (small files on a failure), removal in a `finally`, and the sweep of folders whose owner ended. Every step is a POSIX shell script run through `wsl.exe` (`run_sh`). |
| `chem.eng_xtb`, `chem.eng_tblite`, `chem.eng_mopac` | Plugin engines: the xtb program (GFN2/GFN1/GFN-FF), tblite in process, MOPAC PM7/PM6-D3H4X. |
| `chem.eng_gxtb` | g-xTB single points with the official g-xtb build of xtb (`xtb --gxtb`): its own program entry `gxtb`, the build's SHA-256 against the pinned releases (`KNOWN_BUILDS`), a start check, the command and the energy. Not a refinement engine. |
| `chem.cpcm_worker` | FAIL-1: the rank step's GFN1-xTB gas and CPCM terms in a helper process (one per process, started again after a crash), because tblite 0.8.0's ddX CPCM corrupts the heap at some geometries; JSON lines over stdin/stdout. |
| `chem.rank` | The rank step (RANK-2): `rank_method` levels and the `auto` rules by phase with their fallbacks (`resolve`), single points of the refined conformers at the rank level (`compute`: g-xTB, AIMNet2, the GFN1-xTB CPCM ΔG_solv of tblite), the per-machine single-point cache, the rank window and the molecule's `rank` record. |
| `chem.solvents` | One canonical solvent name mapped to ALPB/GBSA, COSMO (ε) and PCM keywords. |
| `chem.crest` | CREST conformer search into a `ConformerSet` (atom order and topology checked); through WSL it runs in a `chem.wsl_scratch` folder. |
| `chem.search_rule` | CONF-AUTO: which conformer search a molecule gets (`decide`, `crest_for`: "rdkit", "crest", or "auto" = CREST for a neutral molecule with at least `crest_min_rotors` strict rotatable bonds), the charge class and rotor count the advisor uses too, `may_use_crest` for run-level planning, and `merge_pools` (CREST's lowest `refine_top` + RDKit's, duplicates dropped, `ConformerSet.refine_count` so every conformer is refined). |
| `chem.qm`, `chem.psi4_driver` | QM stage: the driver runs inside Psi4's own Python (stdlib + numpy + psi4) on a JSON task; the parent builds `result.qm`, descriptors and the vibrational analysis. |
| `chem.cube`, `chem.surfaces` | Gaussian cubes; density isosurfaces, ESP mapping, GIPF descriptors, point-charge ESP surfaces. |
| `chem.io_cclib`, `chem.spectra_io` | Import other programs' outputs (cclib) and experimental spectra (CSV, JCAMP-DX). |
| `chem.optimize` | ASE LBFGS/FIRE with cancellation. |
| `chem.batch_opt` | Batched L-BFGS and FIRE over `[B, N, 3]` torch tensors, batched dxtb and AIMNet2 models, `refine_batched` (drop-in for `pipeline._refine`), VRAM-based batch sizing with halving on out-of-memory. |
| `chem.vibrations` | Hessian (dxtb autograd or finite differences), Eckart normal modes, IR intensities (km/mol), RRHO/qRRHO thermochemistry (Grimme, *Chem. Eur. J.* 2012, 18, 9955), IR spectra, mode trajectories. |
| `chem.properties` | GFN1-xTB charges, dipole and polarizability at a fixed geometry; GFN2-xTB's polarizability (tblite) for runs whose refinement engine gave none. |
| `chem.estimates` | Cheap estimates in every phase's descriptors step (DESC-2): liquid density and molar volume (Girolami; a packing coefficient on the Bondi volume, fitted to PubChem densities), additive polarizabilities (Bosque-Sales, Wildman-Crippen refractivity), the D4 model's polarizability (optional: PyTorch + tad-dftd4) and Sugden's parachor; no value and a note where a method has no parameters. |
| `chem.liquid` | Dimensionless liquid-state groups from those estimates and the descriptors (DESC-3): Lorentz-Lorenz with Bosque's α and its refractive index, the dipole orientation ratio, packing fraction, sphericity, the Lifshitz Hamaker constant and σ, and kT/v_m^⅔ (the reduced surface tension's unit). |
| `chem.descriptors` | Graph descriptors and 3D ensemble descriptors (best and Boltzmann-weighted). |
| `chem.conformer_compare` | Two programs' conformer sets (Spartan documents or imports, QuantaBench run folders) joined by constitution and matched by a symmetric heavy-atom RMSD that factors out PFAS sibling-group symmetry, plus torsions and stereo; coverage, same-lowest, search vs ranking at a GFN2-xTB common level (tblite), dipole split (`conformers compare`, `cli_conformers`). |
| `chem.export` | Per-molecule writers and batch CSV tables; `read_molecule` reads a molecule's folder back into a `MoleculeResult` (every conformer from the SDF). |
| `chem.pipeline` | `BuildConfig`, `MoleculeResult`, `run_molecule` (the rank step `_rank_step` between the refinement and the descriptors), `run`, `use_batched` (when to batch). |
| `chem.method_guide`, `chem.advisor` | The methods guide as data (entries, what each corrects, the element, charge and spin `COVERAGE` of every method, basis set and CREST level) and the model advisor: `review(inputs, cfg)` profiles a batch (fast path for neutral single-fragment records, a per-process cache), checks coverage, and returns findings tied to docs/VALIDATION.md and three presets with cost estimates (docs/CHOOSING_METHODS.md). |
| `chem.jobs` | Resumable batch runs: `JobStore` (SQLite, WAL), run lock, config guard, hash-based chunks, `run_resumable`, `open_run`, `rebuild_tables`, `merge_tables`; read helpers for the GUI: `molecule_rows` (light rows via SQLite `json_extract`), `stored_tables` (the summary rows and descriptors of built molecules, for a run whose CSVs are missing or incomplete), `load_result`, `load_inputs`, `stored_config`. |
| `analysis.formula` | Formula columns without `eval` (UI-3): `parse` checks an `ast` against a whitelist (numbers, names, `+ - * / **`, unary minus, `log log10 exp sqrt abs min max pow`), `Formula.evaluate` walks the tree with NumPy (NaN in, NaN out), `alias`/`aliases` (identifiers for column names), `fit` (`scipy.optimize.least_squares` of a formula's parameters to a target: values, standard errors, n, MAE, RMSE, R², the fitted formula), `linear_fit`, `parity_stats` (DESC-3). |
| `project` | Project documents (UI-2): the `.qbproj` folder (`project.json`, `molecules.jsonl`, `runs/`, `results/`, `models/`), `save` / `load` with relative "/" paths inside the project and atomic writes, `ProjectError` for another format or a newer version. Plain Python. |
| `analysis.features` | The table's feature presets (DESC-3): `chem.liquid`'s dimensionless groups, ratios of α, μ, V and A and the reduced target σ* as formula templates; `plan` resolves them against the table's columns (a descriptor the run has is reused, a dependency is added first). |
| `analysis.models` | Property models on the table's columns (QSAR-4): `ModelSpec`, `run` (k-fold out-of-fold predictions, leave-one-group-out, a holdout column, the final fit's predictions for every row, the reduced target y/scale multiplied back), `RidgeModel` (NumPy; α by efficient leave-one-out), elastic net / random forest / gradient boosting through scikit-learn when installed, median imputation and standardization inside each fold, `write` (the CSV the table joins and a JSON report). No Qt. |
| `analysis.sheet` | The Analyze table's data without Qt: `ColumnStore` (values per molecule id by column: float64 or object arrays), `Block` and `read_run_tables` (a run folder's summary.csv + descriptors.csv + `jobs.stored_tables`, on a worker thread), `result_values`, `write_csv` / `write_xlsx` (openpyxl, optional) / `tsv`, and `TableState` (column choices, formula columns, plot settings as JSON). |
| `build.surfaces` | Graphite and metal slabs, orientation, placement, overlap report. |
| `md.params` | UFF Lennard-Jones table (Rappé 1992) and graphite presets (AMBER parm94, Steele 1973). |
| `md.lammps` | `write_smd_setup`: `system.data`, `in.smd` (rigid adsorbate, frozen slab, Langevin, `fix smd`), README, JSON; `run_lammps` via the Python module. |
| `md.lammps_log` | `log.lammps` thermo blocks, `pull_force.txt`, detachment work and peak force. |
| `md.oplsaa` | MD-2: OPLS-AA atom typing (alkanes, Watkins-Jorgensen perfluoroalkanes, (fluoro)benzenes) and its parameter table, each value with its GROMACS `oplsaa.ff` line (cross-checked against foyer's XML by `scripts/check_oplsaa_params.py`); RB to OPLS torsion conversion. |
| `md.forcefield` | MD-2: `type_molecule(smiles, ff=)` -> `TypedMolecule` (types, charges, LJ, bonded terms in LAMMPS real units, a source per parameter, LJ overrides); backends `oplsaa` (in process), `gaff2` and `openff` (AmberTools/OpenFF in their own conda env through `md/ff_driver.py`, the Python found next to `antechamber`). |
| `md.liquid` | MD-2: the liquid-slab folder: box plan, Packmol or lattice packing, `system.data`, `ff.params`, `in.bulk` (NPT), `in.slab` (opens the box along z, NVT, pressure tensor and density profile), README, JSON; `run_liquid_slab` via the Python module. |
| `md.surface_tension` | MD-2: σ = (Lz/2)⟨Pzz − (Pxx+Pyy)/2⟩ with block errors, statistical inefficiency, blocking curve; centred density profiles and their tanh fit; bulk density, drift and checks; the LJ tail beyond rc for cutoff variants (Blokhuis et al. 1995); `surface_tension.json` / `.md`. |
| `md.bench` | MD-BENCH-1: this machine's LAMMPS ns/day on a cached liquid C6F14 box per accelerator (OpenMP, GPU package, KOKKOS) and dispersion variant, `lmp -h` and `Performance:` parsing, the CPU/GPU `run 0` pressure check, the registry record (`MachineProfile.md_throughput`), its display and run-time estimates (`md estimate`). |
| `gui.main_window` | Window, docks, toolbars, the Appearance dock, the status line, the 2D/3D splitter and the shared selection; File > Open run folder, Build > Resume run folder. |
| `gui.workspaces` | UI-1's task workspaces: `WORKSPACES` (Build, Calculate, Simulate, Analyze, Library: the docks each shows, by object name and area) and `WorkspaceManager` (the activity bar, switching, each workspace's own `saveState` kept in the window settings, Reset layout). |
| `gui.playback` | `PlaybackBar` under the view: first / previous / play / next / last, a slider, a label and a rate; it holds only an index and a count, so conformers now and frames or molecules later. |
| `gui.library_panel` | The Library dock: `find_run_folders` (run folders under the output folder, read-only, off the GUI thread) and `LibraryPanel` (filter, sort, Open, Resume). |
| `gui.molecule_list` | The Molecules dock for any number of molecules: `MoleculeTableModel` (light rows), `MoleculeFilter` (search, status), `MoleculeListView`, and `ResultStore` (`win.results`: full results read back from run folders on selection, LRU of 12; imports pinned). |
| `gui.table_panel`, `gui.table_dialogs` | The Table dock (UI-3, Analyze): `SheetModel` over the Molecules dock's rows (row fields, posted fields, `analysis.sheet` values, formula columns; its own numeric sort and filter; refreshes only while the Table or Plot dock is in view), `SheetView` (Ctrl+C / Ctrl+Shift+C as TSV), `TablePanel` (selection synced with the Molecules dock, CSV/XLSX export on a worker thread, state in the window settings under `analyze/table` until UI-2), the column chooser, formula and fit dialogs, the Features menu and Join... (`JoinDialog`: columns of a CSV by id, `analysis.sheet.read_join`; DESC-3). It follows `MainWindow.moleculeResult` and `runFolderOpened`. |
| `gui.plot_panel` | The Plot dock (UI-3, Analyze): pyqtgraph scatter of two numeric table columns over the rows the table shows, colour by a third, least-squares line with R², the 1:1 parity view (DESC-3), click a point to select its molecule. |
| `gui.model_dialog` | The Table dock's Model... dialog (QSAR-4): target, model kind, features (presets for the liquid groups and α, μ, V), σ*, folds, seed, group and holdout columns; `analysis.models.run` on a worker thread; Add to table writes into the project's (or the output folder's) models/ and joins the predictions; Plot 1:1. |
| `gui.project_actions` | Projects in the window (UI-2): `ProjectController` (`win.project`): File > New / Open / Open recent / Save / Save as / Close project, the title's `[*]` and the unsaved-changes question (also from `closeEvent`), pinned results written to the project's results/, builds into its runs/ (`MainWindow.run_build`), job-table news and the table's values read on open (`TablePanel.read_tables`). |
| `gui.action_ids` | Named actions and their default shortcuts in one place (`DEFAULTS`, `bind`): the mouse modes, Edit > Undo / Redo, the Model Kit; for a rebinding page (UI-9). |
| `gui.viewer3d` | `MoleculeViewer3D`: PyVista, instanced glyphs, NumPy ray picking, Rotate/Select/Move/Build modes. Build mode draws the free valences it is given as stubs (`set_stubs`) and reports clicks (`buildClicked`); it never edits. `gui.viewer` is a compatibility shim (`MoleculeViewer`). |
| `gui.model_kit` | The Model Kit dock (UI-7): element (common row, periodic table), geometry (`chem.edit.geometry_options`: VSEPR shapes, or coordination geometries for metals), charge and bond pickers, fragments, SMILES / InChI, the edit tools and the controller of Build mode. An open edit owns the 3D and 2D views (as the Surface Builder does; `MainWindow.kit_owns_view`); each edit is a `chem.edit.Command` on a `QUndoStack` (`UndoAdaptor`, Edit > Undo / Redo); Add to list / Replace hand `model.to_input()` to the molecule list. |
| `gui.viewer2d` | `MoleculeViewer2D`: RDKit SVG depiction in a `QGraphicsView`, full-index atom map, in-plane rotation, export. |
| `gui.selection` | `SelectionModel` (ordered atom indices, `changed` signal), measurements, the shared accent color. |
| `gui.spectra_panel`, `gui.surface_panel`, `gui.table_panel`, `gui.plot_panel`, `gui.import_action` | Optional panels installed by `MainWindow._install_panels` (one that fails is logged and left out); they follow `MainWindow.resultChanged` (and `displayChanged` for energy units). |
| `gui.clipboard` | Edit > Paste (ChemDraw CDX, CDXML, MOL, MDLCT, text, in that order; the macOS pasteboard is also read directly for ChemDraw's type) and Copy as SMILES / MOL / CDXML; drops on the window (files, folders, text). |
| `gui.sketcher_dock`, `gui.sketcher_view`, `gui.sketcher_web` | The Sketch dock (UI-6): Schrödinger's Sketcher (WebAssembly) in a `QWebEngineView`, served from a `qb-sketcher://` URL scheme registered before the QApplication; buttons hand MOL V3000 to `chem.sketch` (RDKit checks, `MoleculeInput`). Qt WebEngine is optional and loaded only when the dock opens. `sketcher_assets` fetches the editor's files (pinned sha256) into the programs folder. |
| `gui.collapsible`, `gui.popout` | UI-8: `Section` (a header with an arrow, a one-line summary, a status and an ⓘ over a body that folds), `Accordion` (one section open, remembered), `InfoLabel`/`InfoButton`/`with_info` (a field's notes in a pop-out, ⚠ when something is missing), `Notes` (notes folded to one line); `popout.pop_out`/`dock_all` and View > Pop out panel / Dock all panels. The Pipeline dock's stages, the Molecules dock's details and the Plot's ⧉ use them. UI-8b: the Surfaces and Spectra panels' and Settings > Compute's sections and ⓘ, and `popout.DockTitleBar` (title, ⧉, ✕) on every dock. |
| `gui.results_fold` | UI-8b: the Results dock shrinks to the conformer table's header and row for a molecule with one conformer (or none) and gets its height back with more; View > Fold Results for one conformer. |
| `gui.machine_monitor` | The machine monitor strip in the Pipeline dock, above the Run buttons: this computer (`sensors.Monitor`) and the workers (`RemotePoller`, `/v1/sensors`), sparklines, colours by threshold; a QTimer only copies the threads' latest samples. |
| `chem.recipes` | UI-5's job model, UI-agnostic: `TASKS` (conformer distribution, optimize, single point, frequencies; scan not yet), each a set of `BuildConfig` changes; `Recipe` and the built-in Screen / Standard / Careful (the "gfn2" alias resolved to the installed engine); the user's recipes in `Settings.recipes`; `levels()` by family; `JobSpec` (name, recipe, task, configuration, charge and multiplicity overrides, Run on) and `job_inputs`. |
| `dist.remote_job` | UI-5b / DIST-3: a job sent from the job dialog to a worker and followed on its own thread (`RemoteJob`): submit as one share (`client.job_body`), poll, cancel, then the run folder (the synced copy once its `worker.json` vouches for its tables, else the archive into `from_workers/<worker>/`); `choose_machine` for Run on "auto" (the grid's "auto" weights per waiting job). |
| `gui.job_dialog`, `gui.job_queue`, `gui.job_monitor` | UI-5: the job dialog (Job / Level of theory / Resources / Review pages, Save as recipe..., `submit_spec`); the job queue (every build is a `Job`; `MainWindow.submit_job` hands it over, `_start_job` runs it as before, `_on_finished` / `_on_failed` end it and the next starts); the Jobs dock (status, progress, each job's log, cancel, reorder, open folder, what the running job uses and the machine monitor's latest sample), Build > New job... (Ctrl+J). |
| `gui.review_panel` | The Review window (ADV-1): shown when Run is pressed (`MainWindow.request_run` -> `review_before_run`) and on demand (Review..., F6); the review runs on a worker thread; Apply sets a finding's or preset's settings in the pipeline pages, Explain opens the methods guide. |
| `gui.surface_dialog`, `gui.settings_dialog`, `gui.workers` | Dialogs and `QThread` workers. |
| `gui.energy_widgets` | `EnergySpinBox`: an energy input shown in the display unit and stored in a fixed one (the energy window in kcal/mol), returning the stored value exactly until it is edited. |
| `cli` | `doctor`, `build` (resumable for batches), `status`, `merge`, `freq`, `import`, `slab`, `md smd`, `md parse`, `settings`; `md liquid-slab`, `md surface-tension`, `md estimate` (`cli_md`); `pubchem enrich`/`lookup` (`cli_pubchem`); `conformers compare` (`cli_conformers`); `sketcher` (`cli_sketcher`); `monitor` (`cli_monitor`: live machine metrics); `methods` (`cli_methods`); `review` and the review header of `build` (`cli_review`). |

## Key decisions

- **PySide6, not PyQt6.** LGPL licensing suits a public MIT app, and OVITO's embeddable viewport
  (`ovito.gui.create_qwidget`) is PySide6-only.
- **PyTorch is the GPU abstraction.** AMD ROCm (Windows and Linux), NVIDIA CUDA, and Apple MPS all
  surface as torch devices, so engines written in PyTorch (dxtb, AIMNet2, future MLIPs) inherit GPU
  support. PyTorch is not a hard dependency: users install the build that matches their GPU.
- **Engines are optional and self-describing.** `available()` returns a reason, the GUI shows it, and
  the pipeline falls back (`auto` picks AIMNet2, then GFN2, then GFN1, then MMFF; GPU errors drop to
  CPU) and records each fallback in the molecule's notes.
- **External programs run as subprocesses.** xtb, CREST, MOPAC and Psi4 bring their own OpenMP/BLAS
  runtimes; in-process they would fight PyTorch over threads, and a crash would take the GUI down.
  As subprocesses they can live in their own conda environment (Psi4 must), be cancelled by killing the
  process tree, and run through WSL on Windows. Commands use file names relative to the job folder so
  the same command works natively and in WSL.
- **ASE calculators are the engine contract.** Anything that can be an ASE calculator can plug in.
  Batched engines add a second contract: `energy_and_forces(positions[B, N, 3] Å) -> (E[B] eV,
  F[B, N, 3] eV/Å)`, as a `batch_opt.BatchedModel` (with an optional per-atom charge buffer and
  `supports_subset` for evaluating only unconverged systems). `BuildConfig.batched` = `auto` uses it on
  a GPU, `on` everywhere, `off` never; results match the sequential path.
- **Batches are resumable.** Batches go through `chem.jobs`: every finished molecule is committed to
  `jobs.sqlite` in its own transaction, so a crash loses at most the molecule in progress; a molecule
  that crashes the process twice is marked failed so it cannot block later resumes. Run folders may live
  in a SyncThing folder (Brandon's output folder does), but only one machine may ever *write* a given
  run folder: each machine writes its own (`--chunk`), the others only read it (`status` opens the
  database read-only and never checkpoints it), and `merge_tables` combines the CSVs. Add
  `jobs.sqlite-wal` and `jobs.sqlite-shm` to the SyncThing folder's `.stignore` (see the README) so the
  transient WAL files are never synced. The run lock records host, OS (Windows vs WSL2) and pid.
- **Outputs are plain files.** One folder per run, one subfolder per molecule, a `summary.json` per
  molecule, and CSV tables per batch. Nothing proprietary, and easy to sync (SyncThing) and to read
  from notebooks.
- **Threads are set before heavy imports.** The GUI applies the thread cap before PyTorch or NumPy
  thread pools start. The CLI does it at the start of a run.

## GUI: 2D and 3D views

The central widget is a `QSplitter`: `MoleculeViewer2D` (left, narrower, can be hidden with "Show 2D")
and `MoleculeViewer3D`. The main window creates one `SelectionModel` and passes it to both views.

- **Indices.** The selection holds full-molecule atom indices (explicit hydrogens included): the same
  numbers as RDKit's atom order and ASE's `Atoms`. The 2D view hides hydrogens by default, so it keeps
  a map from depicted atoms to full indices; picking, highlights, index labels and charges go through
  it. A hidden hydrogen selected in 3D simply is not highlighted in 2D.
- **Flow.** A click in either view calls `select`/`toggle` on the model; the model emits `changed`;
  both views redraw their highlights (the 3D view only rebuilds its halo and measurement actors). The
  3D view emits `measurementChanged(str)` for 2-4 selected atoms, shown in the status bar.
- **Showing results.** Selecting a molecule or a conformer row calls `show_rdkit` on both views; the 3D
  view also gets the conformer's charges (for charge coloring and labels). The selection is cleared
  when a different structure is shown (another molecule, or the input preview replaced by the built,
  protonated molecule), since its indices would point at other atoms.
- **Modes.** Rotate / Select / Move act on the 3D view. In Move mode a press grabs atoms as Avogadro's
  manipulate tool does (`grab_group`): on an atom, its whole molecule (connected fragment, or the
  adsorbate on a surface), or the selection if that atom is selected; on empty space, the selection,
  else atoms tagged 1, else everything. A drag rotates the group rigidly about its centroid (Shift:
  slides it); a click does not change the selection; `coordinatesChanged` fires once per drag. Build
  mode (the Model Kit, `gui.model_kit`): a drag still turns the camera, a click is reported as
  `buildClicked` (a stub, an atom or empty space) and hydrogens are drawn as stubs; Ctrl+click selects.
- **Surface builder.** The dialog is modeless and previews in the 3D view (the 2D panel is hidden
  meanwhile). Its "Move adsorbate" button switches to Move mode; `coordinatesChanged` writes the new
  positions into the dialog's system and re-runs the overlap report (and warns if a bond inside the
  molecule changed by more than 0.5 Å), so Save and "Export LAMMPS SMD setup..." use the moved
  geometry. While it is open, selecting another molecule updates only the side panels. The SMD export
  runs on a worker thread owned by the main window, so closing the dialog cannot destroy it.
- **Colors.** One amber accent (`selection.ACCENT_RGB`) marks selected atoms in both views. Charge
  coloring is red = negative, blue = positive on a symmetric scale.

## Threading model (GUI)

`BuildWorker` runs `pipeline.run` (one molecule) or `jobs.run_resumable` (a batch, so an interrupted
GUI batch can be finished with Build > Resume run folder or `quantabench-cli build --resume`) on a `QThread`.
A finished molecule becomes a row of the molecule list (`gui.molecule_list`); selecting one reads its result
from its folder with a `FunctionWorker`, one read at a time. Callbacks become Qt
signals (`progress`, `log`,
`molecule_done`, `finished`). Work done in the GUI process (one lane) runs on that thread lowered to the run
priority (`priority.thread_lowered`); lanes lower their own processes, and the GUI thread and process keep theirs.
Cancel sets a `threading.Event` that is checked between conformers and
at every optimizer step. PubChem lookups, file and Excel reading, hardware detection and the LAMMPS export also run in
workers. Their callbacks must run on the GUI thread: PySide6 runs a closure or lambda connected to a
worker signal in the *worker* thread, so `workers.start(..., on_done=, on_failed=)` routes them through
a relay QObject; connect worker signals directly only to bound methods of GUI-thread QObjects. Closing
never blocks: with work running, the window cancels the build, says "Closing after the current
step...", and closes itself once every worker thread has stopped (a QThread destroyed while running
aborts the process).

## Output layout

```
<output_dir>/<YYYY-MM-DD_HHMMSS>_<label>/
    summary.csv            one row per molecule: status, error, charge as drawn and computed, multiplicity,
                           removed fragments, method, force field, device, best energy, notes
    descriptors.csv        one row per molecule: graph + 3D (best/Boltzmann) + xTB + estimates (density_*, polarizability_*,
                           parachor_sugden) + vib_* (frequencies)
    jobs.sqlite            batch runs: job table (config, inputs, status, per-molecule summaries), and the
                           timings of each run (telemetry_* tables, chem/telemetry.py; quantabench-cli timings)
    jobs.lock              batch runs: host + pid of the process working on the folder (removed on exit)
    <id>/
        <id>.xyz           best conformer
        <id>.extxyz        best conformer with per-atom charges
        <id>_conformers.sdf  all kept conformers with energies and Boltzmann weights
        <id>_ir.csv        IR spectrum (Lorentzian, 20 cm-1 FWHM), with --freq / "spectrum"
        <id>_modes.csv     frequencies, IR intensities, reduced masses, force constants
        <id>_freq.xyz      the tightly re-optimized geometry the frequencies belong to
        <id>_summary.json  everything about this molecule, including notes, timings and vibrations

<smd folder>/              md smd, slab --lammps-dir, or the surface builder
    system.data  in.smd  README.txt  smd_setup.json
    log.lammps  pull_force.txt  smd.lammpstrj  equilibrated.data  final.data   (after the run)

<liquid-slab folder>/      md liquid-slab (MD-2)
    system.data  ff.params  in.bulk  in.slab  in.run  README.txt  liquid_setup.json  packing/
    log.bulk  bulk_density.txt  bulk_final.data                                 (after in.bulk)
    log.slab  slab_box.txt  pressure_tensor.txt  density_profile.txt  slab.lammpstrj  slab_final.data  (in.slab)
    surface_tension.json  surface_tension.md                                    (md surface-tension)
```
