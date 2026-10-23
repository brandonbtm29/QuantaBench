# Architecture

```
            +----------------------- GUI (PySide6) -------------------------+
            |  Molecules dock | 3D viewport (PyVista / OVITO) | Pipeline dock  |
            |  Conformers + Log dock | Settings | Surface builder           |
            +-----------------------------+---------------------------------+
                                          | QThread workers (never block the UI)
            +--------------- CLI (argparse) --------------+
                                          |
                     +--------------------v---------------------+
                     |  chem.pipeline.run(inputs, BuildConfig)   |  UI-agnostic, cancellable,
                     |  progress / log / on_result callbacks     |  per-molecule error capture
                     +--+------+------+-------+-------+------+---+
                        |      |      |       |       |      |
                  inputs  standardize conformers engines properties export
                        \______________________________________/
                                          |
                     +--------------------v---------------------+
                     |  resources: threads, GPU device, memory   |
                     |  settings: JSON in a (synced) folder       |
                     +-------------------------------------------+
```

## Packages

| Module | Responsibility |
|---|---|
| `quantabench.resources` | Hardware detection (CPU, RAM, WSL, PyTorch backend: ROCm/CUDA/MPS), thread caps for BLAS/OpenMP/PyTorch, device selection with memory cap, matmul benchmark. |
| `quantabench.settings` | `Settings` dataclasses, JSON load/save, and the settings-folder pointer (env var, then repo-local pointer, then per-user config). |
| `chem.inputs` | `MoleculeInput` records from SMILES, PubChem names, structure files, and CSV (DTXSID-aware). |
| `chem.standardize` | RDKit cleanup, largest fragment, neutralize, and the pH protonation rule table (or Dimorphite-DL). |
| `chem.conformers` | ETKDGv3 embedding, MMFF94s/UFF, energy-window plus RMSD pruning, Boltzmann weights. |
| `chem.engines` | Engine registry and ASE calculator factory: `DxtbCalculator`, AIMNet2. Element support and availability checks. |
| `chem.optimize` | ASE LBFGS/FIRE with cancellation. |
| `chem.properties` | GFN1-xTB charges, dipole and polarizability at a fixed geometry. |
| `chem.descriptors` | Graph descriptors and 3D ensemble descriptors (best and Boltzmann-weighted). |
| `chem.export` | Per-molecule writers and batch CSV tables. |
| `chem.pipeline` | `BuildConfig`, `MoleculeResult`, `run_molecule`, `run`. |
| `build.surfaces` | Graphite and metal slabs, orientation, placement, overlap report. |
| `gui.*` | Main window, viewer, workers, dialogs, optional OVITO dock. |
| `cli` | `doctor`, `build`, `slab`, `settings`. |

## Key decisions

- **PySide6, not PyQt6.** LGPL licensing suits a public MIT app, and OVITO's embeddable viewport
  (`ovito.gui.create_qwidget`) is PySide6-only.
- **PyTorch is the GPU abstraction.** AMD ROCm (Windows and Linux), NVIDIA CUDA, and Apple MPS all
  surface as torch devices, so engines written in PyTorch (dxtb, AIMNet2, future MLIPs) inherit GPU
  support. PyTorch is not a hard dependency: users install the build that matches their GPU.
- **Engines are optional and self-describing.** `available()` returns a reason, the GUI shows it, and
  the pipeline falls back (`auto` picks AIMNet2, then GFN2, then GFN1, then MMFF; GPU errors drop to
  CPU) and records each fallback in the molecule's notes.
- **ASE calculators are the engine contract.** Anything that can be an ASE calculator can plug in.
  Batched GPU engines will add a second contract (a batch of conformers in, energies and forces out)
  in 0.2.
- **Outputs are plain files.** One folder per run, one subfolder per molecule, a `summary.json` per
  molecule, and CSV tables per batch. Nothing proprietary, and easy to sync (SyncThing) and to read
  from notebooks.
- **Threads are set before heavy imports.** The GUI applies the thread cap before PyTorch or NumPy
  thread pools start. The CLI does it at the start of a run.

## Threading model (GUI)

`BuildWorker` runs `pipeline.run` on a `QThread`. Callbacks become Qt signals (`progress`, `log`,
`molecule_done`, `finished`). Cancel sets a `threading.Event` that is checked between conformers and
at every optimizer step. PubChem lookups and hardware detection also run in workers. On close, the
window waits for running workers.

## Output layout

```
<output_dir>/<YYYY-MM-DD_HHMMSS>_<label>/
    summary.csv            one row per molecule: status, charge, method, device, best energy, notes
    descriptors.csv        one row per molecule: graph + 3D (best/Boltzmann) + xTB properties
    <id>/
        <id>.xyz           best conformer
        <id>.extxyz        best conformer with per-atom charges
        <id>_conformers.sdf  all kept conformers with energies and Boltzmann weights
        <id>_summary.json  everything about this molecule, including notes and timings
```
