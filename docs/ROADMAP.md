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

## 0.2: Throughput, viewer, first Spartan and MD pieces (in progress)

Done (merged on main, tested on Linux CPU):

- [x] **2D + 3D viewer.** RDKit 2D depiction (vector, picking, grab-to-rotate, export) next to the
      PyVista 3D view, sharing one atom selection. Avogadro-style mouse and keys, Rotate / Select / Move
      modes, distance/angle/dihedral readout, element/index/charge labels, element/charge/tag colors,
      instanced rendering for 20k-atom slabs, high-resolution screenshots.
- [x] **Batched refinement within a molecule.** All conformers in one torch batch (dxtb
      `batch_mode=2`, AIMNet2 batched input) with vectorized L-BFGS and FIRE. Identical steps and energies
      (1e-12 eV) to sequential ASE LBFGS; 2.4x faster on 2 CPUs for PFOA x 8 conformers. Default `auto`
      batches only on a GPU.
- [x] **Resumable batch runs.** SQLite job table (`jobs.sqlite`) per run folder, crash-safe commits,
      run lock, config guard, `--chunk k/n` splitting across machines, `status` and `merge`.
- [x] **CompTox / Excel import.** `.xlsx`/`.xlsm`, DTXSID and preferred name kept, SMILES → QSAR-ready
      → MS-ready fallback per row, skipped rows reported.
- [x] **Frequencies and thermochemistry** (pulled forward from 0.3): exact dxtb Hessian with a memory
      guard, finite differences for other engines, Eckart projection, IR intensities, IR spectrum CSV,
      RRHO/qRRHO thermochemistry. `build --freq`, the `freq` command, and a GUI checkbox.
- [x] **LAMMPS steered-MD bridge** (pulled forward from 0.4): rigid adsorbate pulled off a frozen
      slab, UFF and graphite LJ presets, PPPM with slab correction, GPU package flags, log and
      pull-force parsing (detachment work, peak force). `md smd`, `md parse`, `slab --lammps-dir`, and a
      button in the surface builder.

Still open for 0.2:

- [ ] **Verify on the RX 9060 XT under native Windows (ROCm 7.2.1 PyTorch).** Nothing GPU-specific has
      run on real hardware yet. To check:
  - dxtb on GPU, single and batched (`batch_mode=2` in float32: SCF convergence for every member);
  - AIMNet2 on GPU (weights could not be downloaded in the build container, so the AIMNet2 paths,
    batched and Hessian, are tested only against stand-ins): energies against `AIMNet2ASE` to ~1e-4 eV,
    charges summing to the total charge, DFTD3/Coulomb with batch > 1;
  - `torch.cuda.mem_get_info` under Windows ROCm, and calibrate `batch_opt.bytes_per_system` with
    `torch.cuda.max_memory_allocated`;
  - the dxtb Hessian memory estimate (calibrated on CPU float64 only);
  - the 3D viewer frame rate for a 20k-atom slab (measured only on software OpenGL: ~0.4 s/frame);
  - LAMMPS `in.smd` with the GPU package (`neigh no`, `compute group/group` untested on GPU);
  - `doctor --bench` numbers for the docs.
- [ ] **Confirm on the first Windows CI run and on BTM NOVA** what could only be checked by reading the
      code here: the Windows CI job with Mesa's llvmpipe OpenGL (`pyvista/setup-headless-display-action`,
      on-screen QtInteractor tests), and the Windows-vs-WSL2 run lock (`jobs._platform_id`) with a GUI
      batch on `C:` and a `build --resume` from WSL on `/mnt/c`.
- [ ] Batching across molecules (padding, dxtb `batch_mode=1`, AIMNet2 `mol_idx`).
- [ ] Parallel CPU stage: a process pool for embedding and MMFF across molecules while the GPU refines.
- [ ] Validation notebook against the Spartan Essential 76 PFAS results (RMSD, energy ranks,
      descriptors, frequencies).
- [ ] GUI: resume a run folder from the app (today: `quantabench-cli build --resume`; GUI batches
      already write the job table), IR spectrum plot and normal-mode animation in the viewer
      (`vibrations.mode_trajectory` exists), live pull-force plots.

Known science limits to keep in mind: GFN1/GFN2-xTB place C=O and carboxylate stretches far too high
(PFOA anion 2013 vs ~1690 cm⁻¹); the SMD model is vacuum, rigid-adsorbate and non-polarizable, so its
work is a gas-phase lower bound, not an aqueous free energy.

## 0.3: Spartan parity for properties (later)

- [x] Frequencies and thermochemistry (done in 0.2; AIMNet2's autograd Hessian still needs real weights
      to verify).
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
- [ ] LAMMPS input templates: minimize, `nve/limit` relax, NVT/NPT, annealing, and steered MD with a
      flexible adsorbate and solvent. (0.2 has steered MD for a rigid adsorbate on a frozen slab, in
      vacuum, with GPU package flags; Jarzynski averaging over repeated pulls is still to do.)
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
