# QuantaBench: vision and value

## The problem

Molecular modeling on a personal workstation still means stitching together tools that were never
designed to work together:

- **Commercial suites** (Spartan, Materials Studio) are polished, but they are closed, licensed per
  seat, mostly CPU-bound, and awkward to script across thousands of molecules. A Spartan project for
  the EPA Essential 76 PFAS list is a single 600 MB binary file that only Spartan can open.
- **Open libraries** (RDKit, ASE, OpenMM, xTB, PySCF) are excellent, but they have no interface, and
  every project re-implements the same glue: parse identifiers, fix charge states, embed 3D, search
  conformers, optimize, export.
- **Molecular dynamics setup is the hardest step.** Topology, partial charges, overlaps, and box setup
  are where simulations "explode." Hand-written geometry scripts drift from reality (for example, a
  rectangular carbon lattice instead of honeycomb graphite), and generic charges on fluorinated
  molecules blow them apart.
- **Consumer GPUs sit idle.** A 16 GB Radeon or GeForce card can run modern machine-learning
  potentials hundreds of times faster than DFT. Almost no desktop chemistry tool uses one, and AMD
  cards on Windows are usually ignored altogether.

## What QuantaBench is

An open, GPU-aware workbench that takes you from **a name, SMILES, or list of thousands of compounds**
to **simulation-ready, property-annotated 3D structures**, and shows every step in a viewport you can
interact with. It is a front end and orchestrator, not a new physics code: it wires the best open
engines into one reproducible pipeline and keeps them busy on all of your hardware.

## Value, ranked

1. **Near-DFT geometries in seconds, on the GPU you already own.** ML potentials such as AIMNet2
   (trained to ωB97M-D3) and tight-binding (GFN-xTB via dxtb, pure PyTorch) run on AMD ROCm, NVIDIA CUDA,
   or CPU through one device layer. Spartan '26 itself now relies on neural-network estimates
   (DLFFG1 geometries, ωB97X-V/ωB97M-V energy estimates); QuantaBench does the same with open,
   inspectable models.
2. **One reproducible pipeline from identifier to structure.** Standardize, protonate, embed, search
   conformers, refine, compute Boltzmann weights and descriptors, then export. Every molecule gets a
   `summary.json` recording its inputs, settings, engine, device, notes, and timings, and every batch
   gets `summary.csv` and `descriptors.csv`. The same code drives the GUI (exploring a few molecules)
   and the CLI (an overnight run of 4,700 compounds).
3. **PFAS- and separations-aware chemistry out of the box.**
   - Counter-ion stripping (potassium PFOS becomes PFOS).
   - Acidic perfluoroalkane sulfonamides (FBSA, FOSA) deprotonated at pH 7.
   - Engines that handle anions (AIMNet2 and xTB do; several popular ML potentials are neutral-only).
   - QSAR descriptors that matter for fluorinated surfactants: CF2/CF3 counts, F/C ratio, and
     Boltzmann-averaged volume, SASA, and radius of gyration, plus xTB polarizability and dipole
     (the inputs to the "quantum parachor").
4. **A bridge to molecular dynamics.** Optimized geometry plus atomic charges are exported to a
   LAMMPS data file. A surface builder makes correct AB-stacked graphite and places the molecule at a
   controlled gap with an overlap check. Phase 2 adds automatic force-field typing and LAMMPS inputs
   that build in the fail-safes learned the hard way (`fix nve/limit`, `extra/special/per/atom`,
   anchored slab layers).
5. **Use the whole machine, under your control.** One setting caps threads for RDKit, BLAS, and
   PyTorch. GPU use can be set to auto, on, or off, with a device picker and a memory cap.
   `quantabench-cli doctor --bench` reports which engines actually run on your GPU and how fast.
6. **Open and extensible.** MIT-licensed, a small engine interface (an ASE calculator factory), and a
   UI-agnostic pipeline. New ML potentials drop in without touching the GUI.

## Use cases driving the design

### 1. PFAS QSAR at list scale

Today's chain is RDKit filtering, then Spartan (MMFF/PM6 conformers, then ωB97X-D/6-31+G* or Spartan's
neural-network estimates), then a manual descriptor export, then XGBoost and the parachor recalibration.

QuantaBench replaces the middle of that chain with a scriptable, GPU-accelerated stage:

```
quantabench-cli build --file "Cohort 1A Neutral Short.csv" --id-col DTXSID --method aimnet2 --ph 7
```

The output is a `descriptors.csv` with one row per DTXSID, ready to join with EPA/ITRC labels. A
DFT subset (Spartan, or later PySCF) stays as the accuracy anchor.

**Validation target.** Reproduce the Spartan Essential 76 conformer distributions:

- geometry RMSD against Spartan's ωB97X-D structures,
- rank correlation of conformer energies,
- correlation of volume, dipole, and polarizability.

### 2. PFAS on activated carbon, steered MD

1. Build PFOA as the anion, optimized with AIMNet2, with its ML partial charges.
2. Place it 3.5 Å above a three-layer graphite slab with the bottom layer fixed.
3. Export to LAMMPS, run steered MD on the GPU (OpenCL package on native Windows), and view the
   trajectory in OVITO.

Phase 2 automates the force field (OpenFF/GAFF2/OPLS-AA) and generates the input script.

### 3. Teaching and quick exploration

Type a name, see the molecule, its conformers, and their populations, and compare methods
(MMFF vs xTB vs ML) on the same structure in seconds, with nothing to license.

## How it compares

| | Spartan '26 | Avogadro 2 | OVITO | Materials Studio | SEAMM | **QuantaBench** |
|---|---|---|---|---|---|---|
| Open source | no | yes | Basic only | no | yes | **yes (MIT)** |
| Own GUI | yes | yes | yes | yes | yes | **yes (PySide6)** |
| Conformer search + Boltzmann | yes | limited | no | yes | via plugins | **yes** |
| ML potentials | proprietary NN estimates | plugin scripts | no | yes (licensed) | limited | **AIMNet2, xTB, more planned** |
| Consumer GPU (incl. AMD on Windows) | no | no | rendering only | limited | no | **yes, via PyTorch** |
| Batch thousands headless | awkward | no | scripting (Pro) | scripting | workflows | **CLI and same pipeline** |
| MD setup bridge | no | partial | analysis only | yes | yes | **surfaces now, force fields next** |

## Out of scope (on purpose)

- Writing new DFT or MD engines. QuantaBench drives PySCF, OpenMM, and LAMMPS rather than competing
  with them.
- Protein modeling and docking (ChimeraX, PyMOL, and docking suites cover these).
- Replacing OVITO for large-trajectory analysis. QuantaBench embeds OVITO's viewport when it is
  installed.

## Risks and open questions

- **AMD GPU coverage.** dxtb is pure PyTorch and should run on ROCm. The AIMNet2 package uses NVIDIA
  Warp kernels for dispersion and neighbor lists, which may force a CPU fallback on Radeon.
  `doctor --bench` will tell; the pipeline already falls back automatically and says so. A ROCm-safe
  dispersion path is a Phase 1 task.
- **ML potential domain of applicability for PFAS.** Perfluorinated anions are under-represented in
  some training sets. Before trusting large screens, validate against a DFT subset (the Essential 76
  in Spartan).
- **Charges for MD.** ML and xTB Mulliken charges are good for descriptors. Production force fields
  need RESP or AM1-BCC, which is Phase 2.
- **Model licenses.** Weights are downloaded on demand, never bundled. Some models (UMA) carry
  restrictive terms.
- **macOS.** The device layer already understands Apple's `mps` backend. A packaged app and Metal
  testing come later.

## Measures of success

- The Essential 76 PFAS built with AIMNet2 in under 10 minutes on an RX 9060 XT workstation, and the
  4,700-compound CompTox list overnight.
- Median heavy-atom RMSD under 0.3 Å against Spartan's ωB97X-D geometries.
- A PFAS-on-graphite SMD system from SMILES to running LAMMPS in under five minutes of user time,
  with no exploding atoms.
