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
  frequencies + thermo, scan (stub)); **resources** (threads, GPU on/off/device, memory cap).
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

## Needs Brandon (hardware / Windows / decisions)

- **HW-1 · needs-Brandon:** ROCm PyTorch on BTM NOVA; `quantabench-cli doctor --bench` output.
- **HW-2 · needs-Brandon:** dxtb and AIMNet2 on the RX 9060 XT (single + batched); AIMNet2 weights
  download; compare energies CPU vs GPU.
- **HW-3 · needs-Brandon:** GUI on Windows 11 (HiDPI picking, frame rate on a 20k-atom slab, dark mode).
- **HW-4 · needs-Brandon:** LAMMPS `in.smd` with the GPU (OpenCL) build on Windows.
- **DEC-1 · needs-Brandon:** approve `docs/SIMULATE_DESIGN.md` (after RES-1).
- **VAL-1 · needs-Brandon:** Spartan Essential 76 PFAS comparison needs his Spartan exports (SDF/CSV of
  ωB97X-D geometries and properties) placed somewhere the runs can read.

## Later (from ROADMAP)
Surfaces (ESP-colored SAS first), spectra panel with normal-mode animation, properties inspector,
align/overlay, constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
substructure search over results, force-field assignment (OpenFF/GAFF2/OPLS), cross-molecule GPU
batching, process-pool CPU stage.
