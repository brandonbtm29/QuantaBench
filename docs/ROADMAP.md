# Roadmap

Status key: done, next, later. Version numbers are targets, not promises.

## 0.1: Geometry builder (done)

- Inputs: SMILES, names/CAS through PubChem, SDF/MOL/XYZ/PDB/SMI files, and CSV lists with DTXSID
  and SMILES columns.
- Standardization: largest fragment, neutral parent, and pH-dependent protonation (PFAS-aware rule
  table, or Dimorphite-DL).
- Conformers: ETKDGv3, MMFF94s with a UFF fallback, energy window, symmetry-aware RMSD pruning.
- Refinement engines:
  - AIMNet2 (CPU/GPU)
  - GFN1-xTB through dxtb (CPU/GPU, pure PyTorch)
  - GFN2-xTB (Linux/WSL)
  - MMFF
  - automatic GPU-to-CPU fallback
- Properties: Boltzmann weights, graph and 3D descriptors, and GFN1-xTB charges, dipole and
  polarizability.
- Export: XYZ, multi-frame XYZ, extended XYZ with charges, SDF ensemble, PDB, MOL, LAMMPS data, and
  batch CSV tables.
- Surface builder: AB graphite, metal slabs, adsorbate placement with overlap checks.
- PySide6 GUI:
  - OVITO-style layout: viewport, pipeline panel, molecules list, conformer table
  - PyVista rendering
  - optional OVITO viewport
- CLI: `doctor`, `build`, `slab`, `settings`.
- Resource manager (threads, GPU, memory cap) and settings that can live in a synced folder.

## 0.2: Throughput and GPU (next)

- [ ] Verify on the RX 9060 XT under native Windows (ROCm 7.2.1 PyTorch): dxtb on GPU; AIMNet2 on
      GPU, or a ROCm-safe dispersion and neighbor-list path; `doctor --bench` numbers in the docs.
- [ ] Batched refinement: optimize all conformers of a molecule, or many molecules, in one GPU batch
      (AIMNet2's native batching, or TorchSim where it installs) instead of one at a time.
- [ ] Parallel CPU stage: a process pool for embedding and MMFF across molecules while the GPU refines.
- [ ] Resumable batch runs: a job database (SQLite) that skips finished molecules after a crash or
      reboot, needed for the 4,700-compound list.
- [ ] EPA CompTox / ITRC import helpers (DTXSID to SMILES; keep the original identifiers in the
      outputs).
- [ ] Validation notebook against the Spartan Essential 76 PFAS results (RMSD, energy ranks,
      descriptors).

## 0.3: Spartan parity for properties (later)

- [ ] Frequencies and thermochemistry (analytic Hessians from dxtb or AIMNet2 via autograd), with IR
      spectra and scaling.
- [ ] Implicit solvation for xTB (GBSA/ALPB) and solvated ML corrections.
- [ ] Coordinate scans, constrained optimization, transition states (Sella), NEB.
- [ ] Orbitals, density, and ESP surfaces: PySCF single points on the CPU threads, cube grids, PyVista
      isosurfaces and mapped ESP (Spartan's signature visuals).
- [ ] HOMO/LUMO and local ionization potential maps.
- [ ] DFT single points and optimizations through PySCF (CPU), with gpu4pyscf on NVIDIA and a HIP port
      when one exists.
- [ ] More ML potentials: UMA/OMol25 (fairchem), AceFF, MACE-OFF, Egret-1; user choice per run.

## 0.4: Molecular dynamics bridge (later)

- [ ] Force-field assignment: OpenFF Sage and GAFF2 with AM1-BCC charges (openff-toolkit, Interchange);
      OPLS-AA option.
- [ ] Topology export to LAMMPS (`atom_style full` with bonds, angles, dihedrals), OpenMM, and GROMACS.
- [ ] LAMMPS input templates: minimize, `nve/limit` relax, NVT/NPT, steered MD, annealing. Includes the
      fail-safes from the PFAS carbon SMD work and GPU package flags when a GPU is present.
- [ ] Run LAMMPS or OpenMM from the GUI with live plots of temperature, energy, and pulling force from
      the log.
- [ ] OpenMM-ML MD with AIMNet2 or AceFF on the GPU (OpenCL platform on Windows).
- [ ] Packmol-style solvation boxes and amorphous carbon or polymer sorbent builders.
- [ ] Trajectory playback (OVITO viewport, MDAnalysis).

## 0.5: Distribution (later)

- [ ] Windows installer (conda constructor or PyInstaller) with a first-run GPU wizard.
- [ ] Linux packages (conda, AppImage).
- [ ] macOS (Apple Silicon, PyTorch `mps`, Metal rendering).

## Ideas parking lot

- Active-learning fine-tune of an ML potential on PFAS DFT data (closes the domain-of-applicability gap).
- Results database with substructure search (like Spartan's SSPD, built from your own runs).
- Plugin entry points for third-party engines and builders.
- Remote execution on a lab server through Tailscale, with the GUI on a laptop.
