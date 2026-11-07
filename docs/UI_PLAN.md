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
