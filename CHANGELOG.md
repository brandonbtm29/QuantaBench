# Changelog

## 0.2.0 (unreleased)

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
