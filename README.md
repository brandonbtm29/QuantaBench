# QuantaBench

**An open, GPU-aware molecular modeling workbench.** Type a name or SMILES, or load a list of
thousands of compounds, and get simulation-ready 3D structures. Along the way QuantaBench assigns
charge states, searches conformers, refines them with machine-learning potentials or tight binding
on your GPU, computes Boltzmann populations and descriptors, and exports for QSAR or molecular
dynamics. Every step is visible in an OVITO-style 3D workspace.

> Status: **0.1, pre-alpha.** The geometry builder pipeline, GUI, CLI, and surface builder work.
> See [docs/ROADMAP.md](docs/ROADMAP.md) for what comes next and [docs/VISION.md](docs/VISION.md)
> for why this exists.

## What it does today

| Stage | What happens |
|---|---|
| Input | SMILES, chemical names or CAS (PubChem), SDF/MOL/XYZ/PDB/SMI, CSV lists (DTXSID-aware) |
| Standardize & protonate | Largest fragment (strips counter-ions), neutral parent, dominant charge state at your pH. PFAS-aware: carboxylates, sulfonates and acidic perfluoroalkane sulfonamides are handled |
| Conformer search | RDKit ETKDGv3, MMFF94s pre-optimization, energy window, symmetry-aware RMSD de-duplication |
| Refinement | **AIMNet2** (ML, ωB97M-D3 quality, neutral and charged), **GFN1/GFN2-xTB** (dxtb, PyTorch), or MMFF94s, on GPU when available with automatic CPU fallback |
| Properties | Boltzmann weights, volume, SASA, radius of gyration, shape, TPSA, CF2/CF3 counts, F/C ratio, xTB charges, dipole, polarizability |
| Export | XYZ, extended XYZ with charges, SDF conformer ensembles, PDB, MOL, LAMMPS data, batch `summary.csv` and `descriptors.csv` |
| Surfaces | AB-stacked graphite and metal slabs; place a molecule at a set gap with an overlap check |

## Quick start

Install the PyTorch build for your GPU first ([docs/INSTALL.md](docs/INSTALL.md)), then:

```bash
pip install -e ".[all]"
quantabench                       # the desktop app
quantabench-cli doctor --bench    # what hardware and engines QuantaBench sees, and how fast they are
```

Command line:

```bash
# one molecule by SMILES (with an id), refined with AIMNet2 at pH 7
quantabench-cli build "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F PFOA" --method aimnet2

# a whole list from a CSV with DTXSID and SMILES columns, on the GPU, 12 CPU threads
quantabench-cli build --file cohort.csv --id-col DTXSID --gpu on --threads 12

# PFOA on a 39 x 38 Å, three-layer graphite slab, ready for LAMMPS
quantabench-cli slab --molecule "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" -o pfoa_graphite.data
```

## Using all of your hardware

- **CPU:** one thread setting (default: all logical cores but one) caps RDKit, NumPy/BLAS and PyTorch.
- **GPU:** `auto`, `on` or `off`, a device picker, and a memory cap. AMD Radeon works natively on
  Windows through AMD's ROCm PyTorch wheels, and on Linux through ROCm; NVIDIA works through CUDA.
- `doctor --bench` times a matrix multiply and each engine on CPU and GPU, so you know what runs where.

## Engines

| Engine | Package | GPU | Charged molecules | Notes |
|---|---|---|---|---|
| MMFF94s / UFF | RDKit | no | yes | Always available; used for the conformer search |
| GFN1-xTB | dxtb | yes (pure PyTorch) | yes | Also provides charges, dipole, polarizability |
| GFN2-xTB | dxtb + tad-libcint | integrals on CPU | yes | Linux/WSL for now |
| AIMNet2 | aimnet | yes* | yes | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I. *Some kernels use NVIDIA Warp; AMD may fall back to CPU, which `doctor --bench` will show |

## Settings

Settings are a single `settings.json`. Point QuantaBench at a synced folder so every computer shares
them:

```bash
quantabench-cli settings --set-dir "~/SyncThing/QuantaBench/Settings"
```

You can do the same from Settings > Folders in the app.

## Platforms

Windows 11 (native), WSL2, and Linux are the targets. macOS is planned: the device layer already knows
Apple's `mps` backend.

## Built on

[RDKit](https://www.rdkit.org) · [ASE](https://wiki.fysik.dtu.dk/ase/) · [PyTorch](https://pytorch.org) ·
[dxtb](https://github.com/grimme-lab/dxtb) · [AIMNet2](https://github.com/isayevlab/aimnetcentral) ·
[PySide6](https://doc.qt.io/qtforpython-6/) · [PyVista](https://pyvista.org) · optional [OVITO](https://www.ovito.org)

Please cite the underlying methods (GFN-xTB, AIMNet2, ETKDG, MMFF94) when you publish results.

## License

MIT. See [LICENSE](LICENSE). Model weights downloaded at run time carry their own licenses.
