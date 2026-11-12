# UI plan

How QuantaBench's interface is organized and where it is going. The backlog that implements it is
`docs/dev/BACKLOG.md`.

## Principles

- **Its own app, not a clone.** Spartan, Avogadro, OVITO, Materials Studio and others are a feature
  checklist, not layouts to copy. Group commands by what the user is trying to do.
- **General-purpose:** molecular quantum chemistry and ML (geometry, conformers, properties at a chosen
  level of theory) *and* molecular dynamics (system building, force fields, running and analyzing
  simulations).
- **MD ideas come from MD tools** (LAMMPS-GUI, VMD + QwikMD, CHARMM-GUI, OpenMM-Setup, Materials Studio
  Forcite, MAPS, SAMSON, Packmol/moltemplate, OVITO), via the RES-1 survey.
- **Every long operation runs off the GUI thread, can be cancelled, and is resumable.**

## Task-oriented workspaces

A left activity bar switches workspaces. All of them share the 2D/3D viewport, the
molecule/system navigator, the playback bar and the job monitor.

1. **Build:** import (SMILES, InChI, names, CAS, CompTox tables, and files in many formats, with
   Spartan-style import options: UI-4), a 2D sketcher (an embedded open-source editor, Ketcher or
   Schrödinger's Sketcher: UI-6), a Spartan-style 3D model kit (atoms by hybridization, groups, rings,
   PFAS fragments, geometry tools, minimize: UI-7), one undo/redo history for 2D and 3D, isomer/tautomer
   generation, surfaces/slabs/boxes/packing.
2. **Calculate (molecular):** one job dialog built around a *level of theory* and a *task* (see below).
   Presets plus an Advanced section. Conformer distributions, frequencies, constraints/scans,
   transition states.
3. **Simulate (MD):** system → force field → protocol (minimize / equilibrate NVT/NPT / production /
   steered / annealing) → engine (OpenMM, LAMMPS) and resources → live telemetry (T, E, P, pull
   force) → trajectory.
4. **Analyze:** spreadsheet (molecules × properties) with formula columns and plots; spectra (IR, with
   experimental overlay; clicking a peak animates the mode); surfaces (ESP-colored SAS first, orbitals
   later); trajectory analysis (RDF, MSD, density profiles, work/PMF); overlay/align.
5. **Library:** project database of results, substructure search, previous runs and resumable batches.

**The molecule navigator holds any number of molecules** (SCALE-1, done 2026-11-05). Spartan keeps every
molecule of a document rendered in memory, which cut the 14,000-molecule PFAS QSAR set into 200-molecule
documents. QuantaBench's Molecules dock is a virtualized table (`gui.molecule_list`): one light row per
molecule (input order, status, id, conformers, best energy, name, method, formula), sortable by any
column, with a search box (id, name, SMILES, formula) and a status filter. Only the selected molecule and
the last 12 viewed keep their full result in memory; the others are read from their run-folder folder
when selected, off the GUI thread. A batch's finished molecules become rows, not results kept in memory.
**File > Open run folder...** lists a run folder (finished or partial) from its job table, and
**Build > Resume run folder** continues its unfinished molecules with the folder's stored settings. The
Library workspace and the project document (UI-2) build on this.

**The shell is in (UI-1, 2026-11-12):** `gui.workspaces` defines each workspace as a set of docks per area
(the first of a tab group in front) and keeps the user's arrangement per workspace (`saveState`, in the
window settings). The panels are the same objects in every workspace, so a new workspace or a new panel is a
line in `WORKSPACES` plus a dock with an object name. Today's mapping: Build = molecules + Appearance;
Calculate = pipeline, Machines, Appearance; Simulate = the surface builder and LAMMPS export (stub); Analyze =
the Table and Plot (UI-3) in front, surfaces, Appearance, the conformer table and spectra behind their tabs;
Library = run folders under the output folder.

**The spreadsheet is in (UI-3, 2026-11-12):** Analyze's Table dock is the molecules x properties sheet over the
Molecules dock's rows (descriptors from results in memory or a run folder's CSVs), with formula columns (an `ast`
whitelist evaluated with NumPy, never `eval`) and a fit helper for formulas with parameters such as the quantum
parachor; the Plot dock scatters two of its columns. A column whose name is not an identifier is referenced by an
alias the column chooser shows. Until the project document (UI-2) exists, the column choices and formula columns
live in the window settings; UI-2 moves `TablePanel.state()` into the project.

**Shared chrome:**
- A playback bar for conformers, molecules and trajectory frames.
- A status bar with counts, point group, CPU threads, GPU and VRAM.
- A properties inspector: click an atom or bond to see its details.
- One **Appearance** panel for style, labels, colors, hydrogens, H-bonds and R/S labels.

## Structure input (plan 2026-11-07)

Three ways in, all ending in the same `chem.inputs.MoleculeInput` so the CLI and the GUI agree:

- **Text and files (UI-4).** The Molecules box takes SMILES, InChI, InChIKey, names and CAS. File > Open
  reads SDF/MOL, MOL2, PDB, XYZ, SMILES and InChI lists, tables, ChemDraw CDX/CDXML, Marvin MRV, CML, CIF
  and Gaussian/ORCA inputs, and anything else Open Babel reads when the user has it installed. An Import
  Options dialog modeled on Spartan's (2D → 3D, grow hydrogens, post SD fields to the table, filters for
  water, HCl and counter-ions, atom label order) is shown when a file has choices. Ctrl+V pastes from
  ChemDraw and other editors; Copy as SMILES / MOL / CDXML goes back.
- **2D sketcher (UI-6).** A mature open-source editor embedded in a dock (QtWebEngine): Schrödinger's
  Sketcher (BSD-3, RDKit-based), Brandon's choice on 2026-11-07 (Ketcher was the other candidate). It
  hands a MOL block to the pipeline; the 3D view follows. Writing a ChemDraw-class editor from scratch is
  years of work and not the goal.
- **3D model kit (UI-7).** Built into the PyVista view, like Spartan's: pick an atom type with its
  hybridization (or a group, ring or PFAS fragment) and click a free valence to attach it; set bond
  lengths, angles and dihedrals; minimize with MMFF. Every edit is an undoable command on one edit model
  shared with the sketcher.

## Job model: level of theory × task × resources

- **Level of theory**, by family:
  - force field (MMFF94s, UFF; OpenFF/GAFF2/OPLS for MD)
  - tight binding (GFN1/GFN2-xTB via dxtb, tblite or the xtb program; GFN-FF)
  - semi-empirical (PM7, PM6-D3H4X via MOPAC)
  - ML potential (AIMNet2; UMA/OMol25, MACE-OFF, AceFF later)
  - DFT/HF (Psi4 QM stage today: functional + basis; PySCF later for NMR and GPU DFT)
- **Settings:** solvent and charge/multiplicity where supported.
- **Molecular tasks:** single point, optimize, conformer distribution, frequencies + thermochemistry,
  scan, transition state, properties.
- **MD tasks:** minimize, NVT/NPT, production, steered MD, annealing; engine, timestep, length,
  temperature/pressure, force field.
- **Recipes:** saved multi-level protocols such as "Screen: MMFF conformers → GFN2 → AIMNet2 → DFT
  single point". A recipe is a generalization of today's pipeline.
- **Resources per job:** CPU threads, GPU on/off/device, memory cap, and the machine: this one, a
  worker such as BTM NOVA or the MacBook, or auto (DIST-3 in the backlog, PLATFORMS.md). Jobs run in a queue shown in the
  job monitor and can be resumed.
- **The review before submitting (ADV-1, in the app since 2026-11-09).** Brandon (2026-11-09): "a review checking what
  types of atoms and groups are present in the molecules about to be submitted, and which makes recommendations of
  which models to choose from... I didn't know how to choose models or what they were intended to correct."
  - Today it is the Review window (`gui.review_panel`), opened by Run (Build > Review before each run turns that
    off), by Review... next to the Run buttons and by Build > Review molecules... (F6). It never blocks: Run anyway
    starts the run, Cancel run drops it.
  - It shows the batch profile (counts with click-through to the molecules), the methods that can take the batch,
    where each step runs, the findings with icons, Apply (sets the finding's settings in the pipeline pages) and
    Explain (the methods guide at the entry), and the Fast screen / Recommended / Careful presets with estimated
    times.
  - In UI-5's job dialog it becomes the dialog's last page before Submit: the same `chem.advisor.Review`, its
    presets next to the recipes, Apply acting on the dialog's level of theory. The CLI has it as
    `quantabench-cli review` and the header `build` prints.
  - User guide: docs/CHOOSING_METHODS.md.

## Job monitor: live machine metrics (MON-1, 2026-11-09)

Brandon (2026-11-09): "during the submit job page, we could monitor other PC metrics like what the current actual
thread speed is and the temperature of the PC."

- **Today: the machine monitor strip** (`gui.machine_monitor`), in the Pipeline dock just above Run all / Run
  selected / Cancel, so it is in view while a run is set up and while it runs. A "Machine monitor" header folds it
  to one summary line (remembered per machine). One block per machine, this computer first, then every worker that
  answers `GET /v1/sensors` (the MacBook's `quantabench-cli worker`):
  - **CPU** utilization, effective clock (average and the fastest core, as Task Manager's "Speed"), temperature and
    package power;
  - **GPU** utilization, temperature and VRAM used / total (the line is hidden on machines without GPU sensors);
  - **RAM** used / total and percent;
  - sparklines of the last 10 minutes: utilization and temperature for CPU and GPU, RAM percent.
- **Colours:** temperatures amber from 80 °C and red from 90 °C; RAM and VRAM amber from 85 % and red from 95 %; the
  clock amber when the CPU runs below 80 % of its base clock at >= 50 % load (likely throttling). Units always shown
  (%, GHz/MHz, °C, W, GB).
- **Tooltips** say where each number comes from (Windows performance counters, LibreHardwareMonitor, macmon,
  sysfs) and why a missing one is missing; a grey note next to the machine's name says how to get temperatures
  ("brew install macmon", "turn on LibreHardwareMonitor's web server").
- **Never blocks the window:** this computer is sampled by a `sensors.Monitor` thread and the workers by one polling
  thread (every 5 s, 2.5 s timeout; unreachable workers again after a minute, older ones without sensors after ten);
  a QTimer only copies their latest samples. Both stop while the strip is hidden and when the window closes.
- The Machines dock's worker status also shows each worker's latest CPU load, clock and temperature (from
  `/v1/info`).
- **Later, with UI-5's job monitor dock:** the same blocks next to the job list, per-job markers on the sparklines
  (when a stage or molecule started), the run's own CPU share (the run timings' `own_cpu_percent`) against the
  machine's, and a throttling banner from `timings`' rule while a run is going.

## Machines dock: split, budget and pause per machine (DIST-7, 2026-11-11)

Brandon (2026-11-11): "how we balance the machine split and also how many processors / much processing is used ...
include a pause button on a machine and the work is redistributed?"

- **Today** (`gui.machines_panel`): one row per machine (this computer first, then the registry and the workers
  that answer).
  - **Status:** idle, busy, `paused (drain)` or `paused (now)`, "waiting for its grid client after a wake". A worker
    that does not answer shows what its status file last said: `unreachable, paused (last said)`.
  - **Split:** *Auto (measured speed)*, *Manual weight* with a weight box (1 = an average Auto machine) or *Off*.
    This is what `grid build --weights auto` uses, saved per machine at once.
  - **Share:** the molecules of the current molecule list it would get, e.g. "28 of 40 (70 %)". It needs an
    explicit method; Auto uses the last grid of the same protocol, else the benchmark.
  - **Run:** a Pause / Resume button. Its arrow has *Pause now*, which stops the molecules in progress at their
    next checkpoint. This computer is paused through its pause file, which its worker, a grid build's local share
    and this window's own batch follow; a worker through its `POST /v1/pause`.
  - Hardware, throughput (benchmark) and profile age, as before.
- **Processing budget** box under the table, for the selected machine: run priority (default, normal, low,
  background), free threads, a *Max CPU* percent that sets the free threads from its thread count, its lane spec,
  and *Pause its worker while it runs on battery* (on by default). "Save for this machine" writes only that
  machine's entries (a fresh read and write), and says when they apply: the priority at once to running lanes, lanes
  and free threads with the next job.
- **Later:** with UI-5's job dialog, "Run on" picks machines from these rows, and the job monitor shows each share's
  moves (paused, away, work sharing) next to the sparklines. A tray icon on the laptop with Pause / Resume would save
  opening the app.

## Priorities

- **Tier 1:** app shell (UI-1), project document (UI-2), spreadsheet/plots/formulas (UI-3), structure
  input: import formats and options (UI-4), 2D sketcher (UI-6) and 3D model kit (UI-7), job dialog +
  monitor (UI-5), Simulate design (RES-1) then v1.
- **Tier 2:** properties inspector, align/overlay, trajectory player synced with telemetry plots.
  (Done 2026-11-01: the Surfaces panel with ESP-mapped density and vdW/SAS surfaces, and the Spectra
  panel with experimental overlay and mode animation; both are docks that UI-1 will place into the
  Analyze workspace. The pipeline dock gained Conformer search (CREST), Solvent and QM (DFT) pages.)
- **Tier 3:** constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
  substructure search, force-field assignment for full bonded MD.
- **De-prioritized:** protein ribbons/Ramachandran, NOE/pharmacophore points, NMR entry dialogs.
