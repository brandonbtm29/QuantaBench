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

1. **Build:** sketch/edit (2D with live 3D), fragment palette (including PFAS fragments), import
   (SMILES, names, CompTox, files, PDB), isomer/tautomer generation, surfaces/slabs/boxes/packing,
   undo/redo.
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

**Shared chrome:**
- A playback bar for conformers, molecules and trajectory frames.
- A status bar with counts, point group, CPU threads, GPU and VRAM.
- A properties inspector: click an atom or bond to see its details.
- One **Appearance** panel for style, labels, colors, hydrogens, H-bonds and R/S labels.

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

## Priorities

- **Tier 1:** app shell (UI-1), project document (UI-2), spreadsheet/plots/formulas (UI-3), structure
  editor + fragments + undo (UI-4), job dialog + monitor (UI-5), Simulate design (RES-1) then v1.
- **Tier 2:** properties inspector, align/overlay, trajectory player synced with telemetry plots.
  (Done 2026-11-01: the Surfaces panel with ESP-mapped density and vdW/SAS surfaces, and the Spectra
  panel with experimental overlay and mode animation; both are docks that UI-1 will place into the
  Analyze workspace. The pipeline dock gained Conformer search (CREST), Solvent and QM (DFT) pages.)
- **Tier 3:** constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
  substructure search, force-field assignment for full bonded MD.
- **De-prioritized:** protein ribbons/Ramachandran, NOE/pharmacophore points, NMR entry dialogs.
