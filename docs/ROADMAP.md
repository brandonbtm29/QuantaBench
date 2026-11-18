# Roadmap

Status key: done, next, later. Version numbers are targets, not promises.

## Platform and multi-machine track (runs alongside the releases)

QuantaBench targets Windows 11, WSL2, Linux and macOS (Apple Silicon) equally, and should let
Brandon's MacBook and the Windows workstation work on one project together. The details, the platform matrix and
the fork/upstream plan are in [PLATFORMS.md](PLATFORMS.md).

Status 2026-11-18: P1 and P2 are built, and P3's "Run on" routing is in the job dialog. The workstation and the
MacBook ran the PFAS QSAR set together in 15 staged batches (13,206 molecules, gas and water, Nov 11-14), with
machine control (budgets, pause and resume, work moving off a paused machine). CREST runs on the workstation through
WSL. A CREST rebuild of the set's flexible neutral molecules is half done (BACKLOG: CREST-RUN-1). 0.3.0 is tagged
(`v0.3.0`, 2026-11-18).

- **P1 (with 0.3):**
  - macOS in CI (PLAT-2) and the first run on the MacBook (PLAT-1)
  - two machines by hand: `--chunk` + SyncThing + `merge` (M0, works today)
  - GPU engines on both the workstation (ROCm) and the Mac (MPS): AIMNet2 without Warp (FORK-3), device-split
    dxtb linear algebra (FORK-4)
  - GFN2 through dxtb on Windows and macOS (FORK-1)
- **P2 (with 0.4):**
  - machine capability reports and throughput-weighted chunks (DIST-1)
  - `quantabench-cli worker` over Tailscale (DIST-2)
  - tblite Windows wheels (FORK-2)
  - Windows checks of the external engines (ENG-1; CREST through WSL, ENG-2, done)
  - machine control: budgets, pause and resume, a closing laptop (DIST-7, done)
- **P3 (with 0.5):**
  - "Run on" machine routing in the GUI (DIST-3, done) and per-stage routing (DIST-4)
  - the staged-batch scripts as features: a time window per machine, deadlines, batch lists (BATCH-1), and CREST
    lanes by rotatable bonds (SCHED-2)
  - installers and terminal-free launchers with an Update button on all three OSes
  - native CREST on Windows (FORK-5), LAMMPS fixes and a Mac GPU build (FORK-8), OpenMM on Metal (FORK-9)

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

## 0.2: Throughput, viewer, first Spartan and MD pieces (mostly done)

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

- [ ] **Verify on the RX 9060 XT under native Windows (ROCm 7.2.1 PyTorch).** AIMNet2 and dxtb were checked on
      the GPU on 2026-11-05 (GPU-1, GPU-2: AIMNet2 about 10x the CPU; dxtb runs but is slower than the CPU, so it
      is off by default). Still open from the original list: the viewer's frame rate for a 20k-atom slab and
      LAMMPS `in.smd` with the GPU package (HW-4). The original list:
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
- [ ] **Confirm on the first Windows CI run and on the Windows workstation** what could only be checked by reading the
      code here. Done: the Windows CI job with Mesa's llvmpipe OpenGL (CI is green on Windows, Linux and macOS
      since 2026-11-05, FIX-1), and the full suite with the GUI tests on the workstation (HW-3). Still open: the
      Windows-vs-WSL2 run lock (`jobs._platform_id`) with a GUI batch on `C:` and a `build --resume` from WSL on
      `/mnt/c`.
- [ ] Batching across molecules (padding, dxtb `batch_mode=1`, AIMNet2 `mol_idx`).
- [x] Parallel CPU stage: done as parallel lanes on one machine (2026-11-05: MMFF 3.7x, GFN1-xTB 3.45x on the
      workstation; tblite gets one lane per physical core).
- [ ] Validation notebook against the Spartan Essential 76 PFAS results (RMSD, energy ranks,
      descriptors, frequencies).
- [x] GUI: resume a run folder from the app: File > Open run folder... and Build > Resume run folder
      (SCALE-1, 2026-11-05; the list also holds 14,000 molecules now).
- [ ] GUI: live pull-force plots. (The IR spectrum plot and normal-mode animation landed with 0.3's
      Spectra panel.)

Known science limits to keep in mind: GFN1/GFN2-xTB place C=O and carboxylate stretches far too high
(PFOA anion 2013 vs ~1690 cm⁻¹); the SMD model is vacuum, rigid-adsorbate and non-polarizable, so its
work is a gas-phase lower bound, not an aqueous free energy.

## 0.3: Spartan parity for properties (0.3.0 tagged 2026-11-18; the open items below carry over)

The original QuantaBench plan (CREST conformers, GFN2/GFN-FF, MOPAC PM6/PM7, Psi4 DFT with density and
ESP cubes, PyQtGraph spectra, cclib, a conda environment) landed here on 2026-11-01, on top of a shared
external-engine layer (`chem/external.py`, engine plugins). See [ENGINES.md](ENGINES.md).

- [x] Frequencies and thermochemistry (done in 0.2; AIMNet2's autograd Hessian still needs real weights
      to verify).
- [x] **External-engine layer**: program discovery (settings, env vars, PATH, conda envs, `wsl:`), thread
      and stack control, cancellation, scratch folders; engine plugin registry; Settings > Engines tab.
- [x] **Implicit solvation**: one solvent name for ALPB/GBSA (xtb, tblite), COSMO (MOPAC), PCM (Psi4).
- [x] **xtb program** (GFN2, GFN1, GFN-FF) with its own optimizer: GFN2 on native Windows.
- [x] **tblite** in process (GFN2/GFN1, ALPB/GBSA/CPCM, finite-field polarizability).
- [x] **MOPAC** PM7 / PM6-D3H4X with COSMO, heats of formation, POLAR polarizability, COSMO area/volume.
- [x] **CREST** iMTD-GC conformer search (GFN2, GFN-FF, GFN2//GFN-FF), RDKit fallback; WSL on Windows.
- [x] **Psi4 QM stage**: DFT/HF single points or optimizations of the lowest conformers, dipole,
      quadrupole, Mulliken/Löwdin/MBIS charges, HOMO/LUMO, static polarizability (analytic or finite
      field), PCM, frequencies with IR and qRRHO thermochemistry.
- [x] **Density and ESP surfaces**: cube reader, 0.002 au isosurface volume/area, ESP mapping,
      Murray-Politzer GIPF descriptors, point-charge ESP fallback, Surfaces panel.
- [x] **IR spectrum panel**: broadened spectrum + sticks, QM trace, experimental overlay (CSV, JCAMP-DX),
      click a peak to animate its normal mode.
- [x] **cclib import** of other programs' outputs (GUI and `quantabench-cli import`).
- [x] Conda environment files for the external programs (`envs/`).
- [ ] First run of all of the above on native Windows (ENG-1: the tests run on the workstation; MOPAC's
      polarizability differs by build, WIN-2).
- [x] CREST through WSL on Windows (ENG-2, 2026-11-16), and CREST for flexible neutral molecules:
      `--conformers auto` (CONF-AUTO).
- [x] Spartan `.spartan` import (IMPORT-2), validation tooling and the first 300-molecule run (VAL-1).
- [x] Spartan's van der Waals radii (CPK area/volume to 0.5 %/0.1 %), the corrected solvent-accessible area,
      ESP polar areas; Settings > Surfaces & radii and Settings > Display (kJ/mol).
- [ ] Validation against the Spartan Essential 76 PFAS ωB97X-D results (VAL-2: data imported, runs planned).
- [ ] A settings dialog as detailed as Spartan's Preferences (SETTINGS-1: Compute, import filters, Lookups,
      Machines left).
- [x] Structure input like Spartan's: import from many formats with an Import Options dialog and ChemDraw
      clipboard (UI-4), an embedded 2D sketcher (UI-6), the 3D model kit's edit model (UI-7). Left: the model
      kit's build mode in the 3D view, and hand checks on Windows and macOS. Plan in UI_PLAN.md.
- [x] **Conformer ranking** (RANK-1, RANK-2): a rank step after refinement, g-xTB with a GFN1-xTB CPCM term in
      water and AIMNet2 in the gas phase, on by default; chosen after benchmarks against DFT (VALIDATION.md).
- [x] **Run tooling:** run timings in every run folder and `timings` (SCHED-1 step 1), a review of the molecules
      before a run (ADV-1), a live machine monitor (MON-1), run priority and free threads (PRIO-1), machine
      control (DIST-7), density and polarizability estimates (DESC-2).
- [x] **Workspaces and projects:** the app shell with task workspaces (UI-1), project documents (UI-2), the
      table with plots and formula columns (UI-3), a feature builder (DESC-3) and property models (QSAR-4) in the
      Analyze table, collapsible and pop-out panels (UI-8), the job dialog with recipes and a job queue (UI-5),
      and "Run on" a worker (UI-5b).
- [x] **The PFAS QSAR set in staged batches** (QSAR-1): 15 batches, 13,206 molecules, gas and water.
- [ ] Orbital isosurfaces (MO cubes), HOMO/LUMO and local ionization potential maps.
- [ ] NMR shieldings (Psi4 has none: PySCF GIAO, or ORCA through cclib import).
- [ ] Coordinate scans, constrained optimization, transition states (Sella), NEB.
- [ ] GPU DFT: gpu4pyscf on NVIDIA, a HIP port when one exists.
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
- [ ] macOS `.app` bundle (Apple Silicon; notarization later). macOS itself is no longer deferred to
      0.5: it is brought up in P1 of the platform track.
- [ ] On every OS: an Update button (pull/upgrade, then restart) and a launcher that hides the
      terminal window.

## Ideas parking lot

- Active-learning fine-tune of an ML potential on PFAS DFT data (closes the domain-of-applicability gap).
- Results database with substructure search (like Spartan's SSPD, built from your own runs).
- Plugin entry points for third-party engines and builders.
- Remote execution on a lab server through Tailscale, with the GUI on a laptop (now planned as
  DIST-2/DIST-3 for the Windows workstation + MacBook; a lab server would be one more worker).
