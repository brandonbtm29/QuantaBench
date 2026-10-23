# Changelog

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
