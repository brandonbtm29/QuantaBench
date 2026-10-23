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

**Shared chrome:**
- A playback bar for conformers, molecules and trajectory frames.
- A status bar with counts, point group, CPU threads, GPU and VRAM.
- A properties inspector: click an atom or bond to see its details.
- One **Appearance** panel for style, labels, colors, hydrogens, H-bonds and R/S labels.

## Job model: level of theory × task × resources

- **Level of theory**, by family:
  - force field (MMFF94s, UFF; OpenFF/GAFF2/OPLS for MD)
  - tight binding (GFN1/GFN2-xTB; PM6/PM7 via MOPAC later)
  - ML potential (AIMNet2; UMA/OMol25, MACE-OFF, AceFF later)
  - DFT/HF (PySCF: functional + basis + dispersion)
- **Settings:** solvent and charge/multiplicity where supported.
- **Molecular tasks:** single point, optimize, conformer distribution, frequencies + thermochemistry,
  scan, transition state, properties.
- **MD tasks:** minimize, NVT/NPT, production, steered MD, annealing; engine, timestep, length,
  temperature/pressure, force field.
- **Recipes:** saved multi-level protocols such as "Screen: MMFF conformers → GFN2 → AIMNet2 → DFT
  single point". A recipe is a generalization of today's pipeline.
- **Resources per job:** CPU threads, GPU on/off/device, memory cap. Jobs run in a queue shown in the
  job monitor and can be resumed.

## Priorities

- **Tier 1:** app shell (UI-1), project document (UI-2), spreadsheet/plots/formulas (UI-3), structure
  editor + fragments + undo (UI-4), job dialog + monitor (UI-5), Simulate design (RES-1) then v1.
- **Tier 2:** surfaces, spectra panel, properties inspector, align/overlay, trajectory player synced
  with telemetry plots.
- **Tier 3:** constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
  substructure search, force-field assignment for full bonded MD.
- **De-prioritized:** protein ribbons/Ramachandran, NOE/pharmacophore points, NMR entry dialogs.
