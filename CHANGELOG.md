# Changelog

## 0.3.0.dev (unreleased)

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
- `doctor` shows which AIMNet2 and dxtb path each GPU gets; the `ml` extra adds `tad-dftd3`.
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
