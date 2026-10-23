# QuantaBench

**An open, GPU-aware molecular modeling workbench.** Type a name or SMILES, or load a list of
thousands of compounds, and get simulation-ready 3D structures. Along the way QuantaBench assigns
charge states, searches conformers, refines them with machine-learning potentials or tight binding
on your GPU, computes Boltzmann populations and descriptors, and exports for QSAR or molecular
dynamics. Every step is visible in an OVITO-style 3D workspace.

> Status: **0.2 in development, pre-alpha.** The geometry builder pipeline, the 2D/3D viewer, the
> GUI, CLI, surface builder, frequencies and thermochemistry, resumable batch runs and the LAMMPS
> steered-MD bridge work. GPU paths are written but not yet verified on real hardware.
> See [docs/ROADMAP.md](docs/ROADMAP.md) for what comes next and [docs/VISION.md](docs/VISION.md)
> for why this exists.

## What it does today

| Stage | What happens |
|---|---|
| Input | SMILES, chemical names or CAS (PubChem), SDF/MOL/XYZ/PDB/SMI, CSV and Excel lists (CompTox exports: DTXSID, preferred name, SMILES / QSAR-ready / MS-ready fallback) |
| Standardize & protonate | Largest fragment (strips counter-ions), neutral parent, dominant charge state at your pH. PFAS-aware: carboxylates, sulfonates and acidic perfluoroalkane sulfonamides are handled |
| Conformer search | RDKit ETKDGv3, MMFF94s pre-optimization, energy window, symmetry-aware RMSD de-duplication |
| Refinement | **AIMNet2** (ML, ωB97M-D3 quality, neutral and charged), **GFN1/GFN2-xTB** (dxtb, PyTorch), or MMFF94s, on GPU when available with automatic CPU fallback. On a GPU all conformers of a molecule are optimized in one batch (same results as one at a time) |
| Properties | Boltzmann weights, volume, SASA, radius of gyration, shape, TPSA, CF2/CF3 counts, F/C ratio, xTB charges, dipole, polarizability |
| Frequencies | Hessian (exact for xTB, finite differences otherwise), normal modes, IR intensities and spectrum, ZPE / H / S / G (ideal gas, Grimme qRRHO) for the best conformer |
| Export | XYZ, extended XYZ with charges, SDF conformer ensembles, PDB, MOL, LAMMPS data, IR spectrum CSV, batch `summary.csv` and `descriptors.csv` |
| Batches | Resumable: a SQLite job table in the run folder survives crashes and reboots; split a list across machines with `--chunk k/n` and merge the tables |
| Surfaces | AB-stacked graphite and metal slabs; place a molecule at a set gap, grab and turn it with the mouse, overlap check |
| Molecular dynamics | LAMMPS steered-MD setup (adsorbate pulled off a frozen graphite slab), log and pull-force analysis (detachment work, peak force) |
| Viewer | 2D structure (RDKit) and 3D (PyVista) side by side, sharing one atom selection; distance/angle/dihedral readout; element, charge or tag coloring |

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

# a CompTox dashboard export (Excel), split over three machines; this one builds part 2
quantabench-cli build --file pfas_list.xlsx --smiles-pref qsar --method gfn1-xtb --chunk 2/3
quantabench-cli build --resume "~/SyncThing/QuantaBench/Runs/<run folder>"   # after a crash or reboot
quantabench-cli status "<run folder>"
quantabench-cli merge combined/ run_chunk1/ run_chunk2/ run_chunk3/
# Run folders in a SyncThing folder: one machine writes each run folder; `status` only reads it.
# Keep SQLite's transient files out of the sync by adding to that SyncThing folder's .stignore:
#   jobs.sqlite-wal
#   jobs.sqlite-shm

# frequencies, thermochemistry (ZPE, H, S, G) and an IR spectrum CSV
quantabench-cli freq "OC(=O)C(F)(F)F TFA" --method gfn1-xtb
quantabench-cli build "...PFOA" --freq          # the same as part of a build

# PFOA on a 39 x 38 Å, three-layer graphite slab, ready for LAMMPS, plus a steered-MD folder
quantabench-cli slab --molecule "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" \
    -o pfoa_graphite.extxyz --lammps-dir pfoa_smd
quantabench-cli md smd --system pfoa_graphite.extxyz -o pfoa_smd_gpu --gpu   # GPU package flags
quantabench-cli md parse pfoa_smd       # after the run: detachment work and peak force
```

`build` uses the resumable job table whenever it builds more than one molecule or gets `--run-dir`,
`--resume` or `--chunk`; a single molecule runs directly. `--batched auto|on|off` controls batched
conformer refinement (auto: only on a GPU).

## Viewer

The center of the window shows the 2D structure (left) and the 3D model (right). Both views share one
selection: an atom picked in either view lights up in both, and the status bar shows the distance,
angle or dihedral of the last 2, 3 or 4 picked atoms. "Show 2D" (Ctrl+D) hides the 2D panel.

**3D view** (Avogadro/PyMOL conventions):

| Mouse / key | Action |
|---|---|
| Left-drag | Rotate the view (trackball) |
| Middle-drag, Shift+left-drag | Pan |
| Wheel, right-drag | Zoom |
| Click an atom | Select it; click empty space to clear |
| Ctrl+click | Add or remove an atom from the selection |
| Double-click an atom | Rotate about that atom from now on |
| R | Reset the view (isometric, fit) |
| X / Y / Z (Shift: from the other side) | Look down an axis |
| P | Perspective / orthographic |
| Esc | Clear the selection |
| Shift+Esc | Cancel a running build (after the current step) |

Toolbar mouse modes: **Rotate** (Ctrl+1, the default), **Select** (Ctrl+2: drag a box) and **Move**
(Ctrl+3: as in Avogadro's manipulate tool, pressing on an atom grabs its whole molecule, or the
adsorbate on a surface; press on a selected atom to grab only the selection. Drag turns the grabbed
atoms rigidly about their center, Shift+drag slides them in the view plane; a click does not change
the selection). Labels: element, index (0-based, as in ASE and RDKit),
element + index, or charge. Colors: element (Jmol), charge (red = negative, white = 0, blue =
positive, on a scale symmetric about zero, as Spartan and Avogadro draw ESP maps), or tag (surface
gray, adsorbate orange).

**2D view:** click selects, Ctrl+click toggles, left-drag on empty space pans, Shift+drag or
right-drag rotates the drawing in-plane, wheel zooms, double-click or F fits, `[` `]` rotate by 15°,
`+` `-` zoom. Hydrogens are implicit by default (View > Hydrogens in 2D). Screenshot... saves the 3D
view as PNG or the 2D structure as SVG/PNG.

**Surface builder:** Build & Preview shows the slab in the 3D view; "Move adsorbate" switches to Move
mode so you can orient the molecule by hand. Save and "Export LAMMPS SMD setup..." write the moved
coordinates.

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
[PySide6](https://doc.qt.io/qtforpython-6/) · [PyVista](https://pyvista.org) · optional [OVITO](https://www.ovito.org) and
[LAMMPS](https://www.lammps.org)

Please cite the underlying methods (GFN-xTB, AIMNet2, ETKDG, MMFF94) when you publish results.

## License

MIT. See [LICENSE](LICENSE). Model weights downloaded at run time carry their own licenses.
