# Architecture

```
            +----------------------- GUI (PySide6) -------------------------+
            |  Molecules dock | 2D view | 3D view (PyVista) | Pipeline dock   |
            |  Conformers + Log dock | Settings | Surface builder           |
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
                                        batch_opt  vibrations
                        \______________________________________/
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
| `chem.inputs` | `MoleculeInput` records from SMILES, PubChem names, structure files, CSV and Excel (CompTox-aware: DTXSID, preferred name, SMILES-column fallback per row, `ReadReport` of skipped rows). |
| `chem.standardize` | RDKit cleanup, largest fragment, neutralize, and the pH protonation rule table (or Dimorphite-DL). |
| `chem.conformers` | ETKDGv3 embedding, MMFF94s/UFF, energy-window plus RMSD pruning, Boltzmann weights. |
| `chem.engines` | Engine registry and ASE calculator factory: `DxtbCalculator`, AIMNet2. Element support and availability checks. |
| `chem.optimize` | ASE LBFGS/FIRE with cancellation. |
| `chem.batch_opt` | Batched L-BFGS and FIRE over `[B, N, 3]` torch tensors, batched dxtb and AIMNet2 models, `refine_batched` (drop-in for `pipeline._refine`), VRAM-based batch sizing with halving on out-of-memory. |
| `chem.vibrations` | Hessian (dxtb autograd or finite differences), Eckart normal modes, IR intensities (km/mol), RRHO/qRRHO thermochemistry (Grimme, *Chem. Eur. J.* 2012, 18, 9955), IR spectra, mode trajectories. |
| `chem.properties` | GFN1-xTB charges, dipole and polarizability at a fixed geometry. |
| `chem.descriptors` | Graph descriptors and 3D ensemble descriptors (best and Boltzmann-weighted). |
| `chem.export` | Per-molecule writers and batch CSV tables. |
| `chem.pipeline` | `BuildConfig`, `MoleculeResult`, `run_molecule`, `run`, `use_batched` (when to batch). |
| `chem.jobs` | Resumable batch runs: `JobStore` (SQLite, WAL), run lock, config guard, hash-based chunks, `run_resumable`, `open_run`, `rebuild_tables`, `merge_tables`. |
| `build.surfaces` | Graphite and metal slabs, orientation, placement, overlap report. |
| `md.params` | UFF Lennard-Jones table (Rappé 1992) and graphite presets (AMBER parm94, Steele 1973). |
| `md.lammps` | `write_smd_setup`: `system.data`, `in.smd` (rigid adsorbate, frozen slab, Langevin, `fix smd`), README, JSON; `run_lammps` via the Python module. |
| `md.lammps_log` | `log.lammps` thermo blocks, `pull_force.txt`, detachment work and peak force. |
| `gui.main_window` | Window, docks, toolbars, the 2D/3D splitter and the shared selection. |
| `gui.viewer3d` | `MoleculeViewer3D`: PyVista, instanced glyphs, NumPy ray picking, Rotate/Select/Move modes. `gui.viewer` is a compatibility shim (`MoleculeViewer`). |
| `gui.viewer2d` | `MoleculeViewer2D`: RDKit SVG depiction in a `QGraphicsView`, full-index atom map, in-plane rotation, export. |
| `gui.selection` | `SelectionModel` (ordered atom indices, `changed` signal), measurements, the shared accent color. |
| `gui.surface_dialog`, `gui.settings_dialog`, `gui.workers` | Dialogs and `QThread` workers. |
| `cli` | `doctor`, `build` (resumable for batches), `status`, `merge`, `freq`, `slab`, `md smd`, `md parse`, `settings`. |

## Key decisions

- **PySide6, not PyQt6.** LGPL licensing suits a public MIT app, and OVITO's embeddable viewport
  (`ovito.gui.create_qwidget`) is PySide6-only.
- **PyTorch is the GPU abstraction.** AMD ROCm (Windows and Linux), NVIDIA CUDA, and Apple MPS all
  surface as torch devices, so engines written in PyTorch (dxtb, AIMNet2, future MLIPs) inherit GPU
  support. PyTorch is not a hard dependency: users install the build that matches their GPU.
- **Engines are optional and self-describing.** `available()` returns a reason, the GUI shows it, and
  the pipeline falls back (`auto` picks AIMNet2, then GFN2, then GFN1, then MMFF; GPU errors drop to
  CPU) and records each fallback in the molecule's notes.
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
  slides it); a click does not change the selection; `coordinatesChanged` fires once per drag.
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
GUI batch can be finished with `quantabench-cli build --resume`) on a `QThread`. Callbacks become Qt
signals (`progress`, `log`,
`molecule_done`, `finished`). Cancel sets a `threading.Event` that is checked between conformers and
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
    summary.csv            one row per molecule: status, charge, method, device, best energy, notes
    descriptors.csv        one row per molecule: graph + 3D (best/Boltzmann) + xTB + vib_* (frequencies)
    jobs.sqlite            batch runs: job table (config, inputs, status, per-molecule summaries)
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
```
