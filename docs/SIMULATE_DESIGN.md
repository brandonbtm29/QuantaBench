# Simulate workspace design (RES-1)

Status: **approved, 2026-11-23** (Brandon: "Yes you can start building out the simulate workspace. For the MD, I would
like to eventually have some default geometries and common set ups"). SIM-1 PR 1 (MD runs as jobs) is built: section
12 says what it does and where it departs from this design. Brandon answered several of the open questions on
2026-11-22; his answers are marked "(Brandon, 2026-11-22)". The default geometries and common setups he asked for,
with the pre-flight checks that go with them, are section 15. Sources are numbered at the end
([Sources](#sources)); repository facts point at the module or file.

The Simulate workspace is where QuantaBench sets up, runs and analyzes molecular dynamics:
system → force field → protocol → engine and resources → live telemetry → trajectory and analysis
([UI_PLAN.md](UI_PLAN.md), "Task-oriented workspaces"). This document surveys how other MD front ends do those six
things, takes stock of what QuantaBench and the November MD pilots already have, and proposes the workspace, its
data model, its engines and force fields, and the order to build it in.

## 1. The recommendation in short

- **Keep the folder as the product.** Every MD setup is already a plain folder that runs with `lmp` alone
  (`md.liquid`, `md.lammps`). The workspace writes, runs, follows and analyzes such folders; it never holds state
  the folder doesn't.
- **Runs become resumable jobs of their own kind**, made of *segments*. A segment is one engine process. Pause,
  time windows, a move from CPU to GPU, or a crash end a segment, and the next one continues from the restart file.
  The pilots needed this by hand.
- **LAMMPS first, OpenMM second.** LAMMPS runs every setup QuantaBench writes, on AMD, Apple and NVIDIA. OpenMM
  comes in v2 for ML-potential MD (OpenMM-ML) and the OpenFF-native path. Since OPENMM-1 (2026-11-23) OpenMM also runs
  liquid slabs as a second engine, much faster on GPUs at reduced precision; LAMMPS stays the reference.
- **Force fields are composed, like physics in COMSOL** (Brandon, 2026-11-22). Each component gets a force-field
  node with a resolution, and global "physics" nodes set the LJ cutoff (the screening length), long-range dispersion
  and electrostatics. Compatibility is checked, defaults are backed by benchmarks, and every parameter keeps its
  source (section 9.1). OPLS-AA stays the reference for perfluoroalkanes; OpenFF Sage (with NAGL charges later) is
  the general route; GAFF2 needs a fluorocarbon refit before it is used for PFAS liquids.
- **σ by default: 16 Å LJ cutoff + tail correction, on the GPU** (Brandon, 2026-11-22). Ewald dispersion on the CPU
  stays as the check. Brandon later the same day: options for the user, with recommendations based on the setup;
  MD-LR's `md.long_range.recommend_long_range` gives 16 Å + tail where LAMMPS can use a GPU and converged Ewald
  dispersion on a CPU-only machine or for a reference value. The C6F14 target is Freire 2006 (12.33 mN/m); OPLS-AA's shortfall is documented, not corrected.
- **Splitting a calculation across machines waits for a consistency test** with pilot A (section 6). v1 runs on this
  computer; machine splitting is v2.
- **The queue schedules by resource** (SCHED-3, section 7): a GPU-bound job and a CPU-bound job run side by side, and
  the user is told so.
- **SIM-1 v1 is two PRs:**
  1. MD runs as jobs: a run supervisor with segments, resume and live telemetry, plus the CLI.
  2. The Simulate panel for the two setups that exist today, the liquid slab and the rigid steered pull, run on
     this computer, with a minimal force-field builder.

  Workers, splitting, OpenMM, mixtures and the extra analyses come in v2 and v3.
- **Default geometries and common setups** (Brandon, 2026-11-23): six setup templates and ten pre-flight checks,
  section 15 (SIM-2, SIM-3).

## 2. What QuantaBench has today

**MD code** (the `md` package; [ARCHITECTURE.md](ARCHITECTURE.md)):

| Piece | What it does | From |
|---|---|---|
| `md.lammps` | Rigid adsorbate pulled off a frozen graphite slab: UFF + graphite LJ, PPPM with slab correction, `fix smd cvel`, vacuum; `system.data`, `in.smd`, README, JSON | 0.2 steered-MD bridge |
| `md.lammps_log` | Thermo blocks of `log.lammps`, `pull_force.txt`, detachment work, peak force | 0.2 |
| `md.forcefield`, `md.oplsaa`, `md/ff_driver.py` | SMILES → `TypedMolecule` (types, charges, LJ, bonded terms, a source per parameter, LJ overrides). OPLS-AA in process; GAFF2 and OpenFF Sage 2.3.0 with AM1-BCC in a conda env through a driver | MD-2 (PR #33) |
| `md.liquid` | Liquid-slab folder: Packmol or lattice packing, `in.bulk` (minimize, warm-up, NPT), `in.slab` (open the box, NVT, pressure tensor, density profile), restart files every 1 ns, GPU switches, pair-energy guard | MD-2 |
| `md.surface_tension` | σ by Kirkwood-Buff with block errors, density profiles and tanh fit, checks, LJ tail beyond the cutoff | MD-2, PR #37 |
| `md.bench` | ns/day per accelerator and dispersion variant, the CPU/GPU `run 0` check, `md_throughput` in the machine profile, `md estimate` | MD-BENCH-1 (PR #35) |
| `cli_md` | `md smd`, `md parse`, `md liquid-slab`, `md surface-tension`, `md estimate`; `md run`, `md status`, `md pause`, `md resume` (SIM-1) | |
| `md.sim`, `md.supervisor`, `md.telemetry` | MD runs as jobs: `run.json` (system, parameterization, protocol parts and stages, segments), the run supervisor (one LAMMPS process per segment, pause through a STOP file, resume from a checkpoint, `progress.json`), the live log reader | SIM-1 PR 1 (2026-11-23) |

**GUI.** The Simulate dock is a stub. Its text, a Surface builder... button and Open in OVITO
(`MainWindow._build_workspace_docks`) are all it has. Shared pieces it can reuse:

- the 3D viewer (instanced, 20k atoms) and the playback bar, which holds only an index and a count, so it can
  step through frames too;
- pyqtgraph plots;
- the collapsible sections and ⓘ notes (UI-8);
- the Jobs dock and the machine monitor (MON-1).

**Jobs and machines.** Every one of these is built around molecules:

- the job dialog's `JobSpec` and `BuildConfig` (UI-5);
- the queue, which runs one build at a time inside the GUI process;
- the worker API, which accepts only `BuildConfig` fields;
- `jobs.sqlite`, which records one row per molecule.

The machine machinery carries over as it is: lanes, run priority and free threads, budgets (Max CPU), time windows
(BATCH-1), pause with drain or now (DIST-7), "Run on" this computer, a worker or Auto (UI-5b), learned rates, and the
MD benchmark's ns/day per machine.

### What the MD pilots taught (2026-11-20 to 22)

Pilot runs: `Runs/md-pilot-2026-11-20` in Brandon's output folder: C6F14 with OPLS-AA, 489 molecules, 2 ns settle and
10 ns production, differing only in the dispersion. The reports of pilots A, B and C and their summary are dated
2026-11-21 and 22. Each lesson below comes with what the design does about it.

| # | Lesson | Consequence for the design |
|---|---|---|
| 1 | On the AMD RX 9060 XT (OpenCL), LAMMPS' default "AMD_GPU" tuning built empty neighbor lists: no nonbonded forces, no error. `device_type generic` fixes it (`md.liquid.gpu_switches`). | An engine *variant* carries its switches. Every input keeps the pair-energy guard. A new GPU, driver or build gets the `run 0` CPU/GPU check (`md.bench`) before a production run. |
| 2 | The fastest setup was `-sf hybrid gpu omp -pk gpu 1 device_type generic pair/only on -pk omp N`: 15.2 against 10.2 ns/day with everything on the GPU (9,780 atoms). Pilot B's slab ran at 21.0 ns/day this way, 5.4 times its 3.9 on 4 CPU threads. | The variant is chosen from measured ns/day, not guessed. Real runs report their speed back as a learned rate. |
| 3 | `lj/long/coul/long` and `pppm/disp` have no GPU styles. | The protocol's dispersion choice limits the engine variants. The review says why the GPU is greyed out. |
| 4 | The dispersion treatment decides σ. Against Tsige & Grest's 11.25 mN/m ([46](#sources)):<br>• C, 12 Å cutoff: 7.96 raw, 11.12 with the tail; slab 4.7 % below the bulk density; 16.1 ns/day on the GPU.<br>• B, 16 Å (Tsige & Grest's setup): 9.60 raw, 11.71 ± 0.30 with the tail, which reproduces their number (+4 %); slab 2.0 % below; 21 ns/day on the GPU + CPU.<br>• A, Ewald dispersion: 10.70 ± 0.70 with no correction needed; slab at the bulk density; 7.6 ns/day on 8 CPU threads.<br>LAMMPS' own tail correction assumes a homogeneous bulk and must not be used for a slab ([40](#sources)). | Cutoff variants always report raw σ, σ with the tail, and the plateau-vs-bulk density check, and the template labels them that way. "16 Å + tail on the GPU" is the default (Brandon, 2026-11-22); "Ewald on the CPU" stays as the check. |
| 5 | Long runs need restart, resume and appended outputs. Pilot B moved from CPU to GPU at 3 ns through a hand-written resume input (`read_restart`, outputs cut at the restart step and appended, `run ... upto`). The join was checked: one continuous pressure-tensor series with no duplicate or missing steps, the same energies and pressures to the fifth digit, and the header lines written again at the join (the reader skips them). The MacBook's GAFF2 slab stopped at its window's deadline. The analysis could not read pilot B's slab speed, because its log was split. | Segments are first-class (section 5). Resume is a button. Outputs append, and every reader skips repeated headers. Telemetry and speed are kept per segment. |
| 6 | The MacBook runs conda-forge LAMMPS (22 Jul 2025) with OpenMP, with restart files every 100 ps, `caffeinate` and a deadline script. | Each machine finds its own engine build. The restart interval follows the machine's window. A Mac GPU build of LAMMPS is FORK-8. |
| 7 | GAFF2 + AM1-BCC gave too high a density for C6F14: 1.836 ± 0.001 against about 1.67-1.68 g/cm³, 10 % high (MacBook bulk NPT, 2026-11-21). | Force-field choice per class, with a warning on fluorocarbons. Refits (Träg & Zahn ([47](#sources))) go through the existing LJ overrides. |
| 8 | OPLS-AA typing covers only alkanes, perfluoroalkanes and (fluoro)benzenes. | Coverage is checked before anything is written, with the alternatives offered. |
| 9 | The pilots ran beside the paused CREST rebuild at below-normal priority on a thread budget. | MD jobs use the same priority and budget settings as builds. A GPU is a resource with its own lane (section 7). |
| 10 | Pilot A's error bar (0.70) missed the 0.5 target, and its blocking curve kept rising. LAMMPS chose a 2 x 2 x 8 dispersion mesh for `force/disp/kspace 0.002`, which may not resolve the interface; a `rerun` of the saved frames with a finer mesh is the proposed check. | The σ result shows the blocking curve and says when it has not levelled off. The README and the review report the PPPM grids LAMMPS chose. Re-evaluating saved frames (`rerun`) becomes an analysis (v2). |
| 11 | Every complete estimate (A, B + tail, C + tail) is 10.7-11.7 mN/m: on Tsige & Grest and the older experiments (11.25-11.44), 5-13 % below Freire 2006 (12.33). | The gap to Freire is the force field, not the method. The C6F14 target stays Freire 2006 with no offset: the OPLS-AA shortfall is documented, not corrected (Brandon, 2026-11-22). Each result is shown against every reference the system has. |

## 3. Survey: how other MD front ends do it

### 3.1 Comparison

Six aspects: **build** (system building), **FF** (force-field assignment), **protocol** (setup), **run** (run
control), **live** (monitoring), **analysis** (trajectory analysis). "—" means the source describes nothing for
it. Commercial tools are described from their vendors' pages, which list features without detail.

| Tool | Build | FF | Protocol | Run | Live | Analysis |
|---|---|---|---|---|---|---|
| LAMMPS-GUI | — (an input editor and runner) | — (whatever the input says) | Hand-written input with syntax highlighting, completion, a variables dialog ([1](#sources)) | LAMMPS in a thread; Stop finishes the current step; progress bar for the current run ([3](#sources)) | Output window with warnings flagged; Charts of thermo columns, live, smoothed, exportable; `dump image` slide show ([2](#sources)) | Plots of imported CSV/YAML data ([1](#sources)) |
| VMD + QwikMD ([7](#sources)) | PDB load, structure checks, mutations, protonation, membranes, water box or implicit solvent, ions ([5](#sources)) | CHARMM36 only ([5](#sources)) | Easy (MD, steered MD) and Advanced (minimization, annealing, equilibration, MDFF, custom templates, NAMD files) ([5](#sources)) | NAMD | Live view through NAMD's interactive MD: pause, detach, finish ([5](#sources), [6](#sources)) | RMSD, RMSF, energies, T/P/V, H-bonds, SASA, SMD forces ([5](#sources)) |
| CHARMM-GUI (web) | Multicomponent Assembler: proteins, membranes, nanomaterials, polymers under PBC ([8](#sources)). Polymer Builder: 60+ monomers, melts and solutions, coarse-grained then all-atom ([9](#sources)) | CGenFF for polymers ([9](#sources)) | Equilibration inputs; melts annealed from Tm + 50 K to 298 K ([9](#sources)) | Inputs for CHARMM, GROMACS, NAMD, LAMMPS, AMBER, GENESIS, OpenMM, Desmond; you run them yourself ([9](#sources)) | — | — |
| OpenMM-Setup | Input files, PDB cleanup ([10](#sources)) | — (not described in its README) | Simulation options ([10](#sources)) | Saves a script or runs it at once; a local browser app ([10](#sources)) | — | — |
| Materials Studio (Amorphous Cell, Forcite Plus) | Polymers, glasses, mixtures by molar ratio; solvent soaking; confined layers for interfaces; target density or a ramp to it; close-contact and ring-spearing checks ([14](#sources)) | COMPASS, cvff, pcff, Dreiding, Universal; charges from the FF, QEq or Gasteiger ([13](#sources)) | Energy, optimization, dynamics (NVE/NVT/NPT/NPH), anneal, quench, cohesive energy density, mechanical properties, confined shear ([13](#sources)) | Interactive calculation ([13](#sources)) | Graphical animation of the run ([13](#sources)) | RDF, MSD and diffusion, concentration/temperature/velocity profiles, density fields, VACF, energy breakdown ([13](#sources)) |
| MAPS (Scienomics) | "Build the digital twin ... at the desired scale" ([15](#sources)) | Force fields with a LAMMPS plugin ([16](#sources)) | Protocols that combine tasks and repeat over many models; Python scripting ([15](#sources)) | Writes all inputs for its engines; "full control of job execution" ([15](#sources)) | — | Analysis toolbox that loads engine outputs; zoom between scales ([15](#sources)) |
| SAMSON | (general modeling platform with extensions; Python API) ([17](#sources)) | GROMACS Wizard: parameters for molecules and force fields ([18](#sources)) | GROMACS Wizard: minimization, NVT/NPT equilibration, production ([18](#sources)); OpenMM Wizard edits OpenMM parameters ([19](#sources)) | Local or cloud GROMACS (bundled 2021.3) ([18](#sources)); OpenMM on CUDA or HIP, or a saved script ([19](#sources)) | — | "analyze results" (no detail) ([18](#sources)) |
| Packmol / moltemplate | Packmol: molecules packed in regions (box, sphere, cylinder, plane, ellipsoid) at a safe distance; Packmol.jl adds PBC and threads ([20](#sources)). moltemplate: `.lt` templates, arrays, polymers from monomers, coordinates from Packmol ([21](#sources)) | moltemplate: templates inherit a force-field object (OPLS-AA documented); other FFs may need charges by hand ([21](#sources)) | moltemplate writes the input script ([21](#sources)) | — | — | — |
| OVITO | — | — | — | — | — (reads trajectories) | Modifiers: RDF, displacements, unwrap, clusters, formulas, smoothing; Pro: spatial binning, time series, diffusion coefficient, Python modifiers ([22](#sources)). Embeddable PySide6 viewport ([23](#sources)) |
| OpenFF Interchange + NAGL (extra) | — (takes an OpenFF Topology) | SMIRNOFF (Sage); NAGL graph-network charges from the molecular graph, production model `openff-gnn-am1bcc-1.0.0` ([25](#sources), [26](#sources)) | — | Exports to OpenMM, GROMACS, LAMMPS, Amber, PDB ([24](#sources)) | — | — |
| RadonPy (extra) | Polymers (homo, random, block, branched) in amorphous, mixture, solution and crystalline cells ([27](#sources)) | GAFF2, GAFF2_mod, Dreiding and others; DFT charges via Psi4 ([27](#sources)) | Automatic equilibration with a completion check and restart scheduling ([27](#sources)) | LAMMPS and Psi4, fully automatic ([27](#sources)) | — | 62 amorphous-state properties (density, Cp, thermal conductivity, Tg, ...) ([27](#sources)) |

Also noted: CHAPERONg, a command-line front end to GROMACS. Its workflow has 12 steps, and a run can start or resume
from any step; it also has steered MD and umbrella-sampling workflows ([28](#sources)).

### 3.2 What to borrow

1. **A folder or script that runs without the front end.** OpenMM-Setup and SAMSON's OpenMM Wizard save a script
   ([10](#sources), [19](#sources)). CHARMM-GUI hands over engine inputs ([9](#sources)). QuantaBench already
   writes runnable folders with a README and a JSON of every setting; keep that as the rule.
2. **Easy and Advanced** (QwikMD ([5](#sources))). A template exposes a few choices; Advanced shows the generated
   input and lets the user save it as a template of their own.
3. **A record of the steps and what to cite** (QwikMD's `.qwikmd` input and `.infoMD` log ([5](#sources))). In
   QuantaBench this is the README and the setup JSON, which list every parameter with its source.
4. **Live charts with smoothing and export** (LAMMPS-GUI ([2](#sources))). Add a **Stop that ends cleanly**, which
   in LAMMPS-GUI finishes the current step ([3](#sources)), and a **live view of the system** (QwikMD
   ([5](#sources))). In QuantaBench the live view is the latest dumped frame, not interactive MD.
5. **Build to a target density, or ramp to it, with contact checks** (Amorphous Cell ([14](#sources))). Pack with
   geometric constraints (Packmol ([20](#sources))): QuantaBench packs at 0.95 × the estimate and lets NPT close
   the gap.
6. **The standard analysis set** (Forcite: RDF, MSD, concentration profiles ([13](#sources))), computed from the
   engine's own output where possible.
7. **One protocol over many systems** (MAPS ([15](#sources)), RadonPy ([27](#sources))). For QuantaBench this is
   the 17 fluorinated test liquids of the MD plan.
8. **Resume from any step and schedule restarts** (CHAPERONg ([28](#sources)), RadonPy ([27](#sources))).

### 3.3 What to avoid

- **Web services that upload structures as a required step** (CHARMM-GUI). Unpublished PFAS stay local by default.
  LigParGen is the exception Brandon allowed (2026-11-22): the review says when a SMILES leaves the machine.
- **Biomolecular assumptions:** PDB residues, one force field (CHARMM36 in QwikMD ([5](#sources))), water as the
  only solvent.
- **The input file as the main interface** (LAMMPS-GUI's model). It stays available as Advanced, not the default.
- **Copying Materials Studio's builders wholesale.** They are years of work; build the systems the PFAS work needs.

## 4. The workspace flow

The Simulate workspace keeps the shared chrome: the activity bar, the 3D view with the playback bar, the Jobs dock.
On the right is a **stepper**: six collapsible sections (UI-8 `Section`s). Each section has a one-line summary and a
status (✓ done, ⚠ check, ● running). The left dock lists the project's **systems** and their **runs**.

```
+--+---------------------------------------------------------------------------------------------------+
|▣ | Systems & runs        | 3D view: the system, or the latest frame of a run    | 1 System        ✓  |
|▣ | ▾ C6F14 slab (OPLS-AA)|                                                      |   C6F14 x 489, slab|
|▣*|     run 1  done  σ    |                                                      | 2 Force field   ✓  |
|▣ |     run 2  ● 7.4/13 ns|                                                      |   OPLS-AA (W&J)    |
|▣ | ▸ C6F14 slab (OpenFF) |                                                      | 3 Protocol      ✓  |
|  | ▸ PFOA on graphite    |                                                      |   Liquid slab (σ)  |
|  | [+ New system]        |  |◀ ◀  ▶  ▶▶|  frame 74 / 120          ns 7.4        | 4 Engine        ✓  |
|  |                       |                                                      |   LAMMPS gpu-cut16 |
|  |                       |                                                      | 5 Telemetry     ●  |
|  |                       |                                                      | 6 Analysis         |
|  +-----------------------+------------------------------------------------------+--------------------+
|  | Telemetry: [T] [P] [E] [ρ] [σ running] [pull force]   7.4 / 13 ns · 21 ns/day · ETA 6 h · GPU 85% |
+--+---------------------------------------------------------------------------------------------------+
```

**1. System.** What is simulated and in what box.

```
Kind      (•) Pure liquid slab   ( ) Bulk liquid   ( ) Molecule on a surface   ( ) Mixture / solution (v2)
Molecule  C6F14  [from the Molecules dock]       molecules 489 (from the size)   atoms 9,780
Density   1.70 g/cm³  QuantaBench estimate ⓘ    [use measured: ____ ]
Box       40 x 40 Å lateral · 100 Å slab · 50 Å vacuum each side       size (•) pilot ( ) full ( ) custom
Packing   Packmol (found) · tolerance 2.0 Å · seed 20261120 · at 0.95 x the density, NPT closes the gap
Checks    ✓ closest contact 2.1 Å   ✓ neutral   [Preview in 3D]
```

"Molecule on a surface" opens today's surface builder (graphite or a metal slab, the molecule placed and moved in the
3D view), and its result becomes the system.

**2. Force field: the parameterization builder** (section 9.1). A tree of nodes, one per component plus the global
physics; v1 has one component node (the liquid, or the rigid adsorbate and its slab).

```
Components
  C6F14 x 489            [OPLS-AA + Watkins-Jorgensen ▾]  [all-atom ▾]  charges [OPLS-AA ▾]  ✓ 20/20 typed
Physics
  Van der Waals          LJ 12-6 · cutoff (screening length) [16 Å]  long range [tail correction, added to σ ▾]
  Electrostatics         PPPM 1e-5
  Combining, 1-4         geometric · 0.5 / 0.5 (from OPLS-AA)
Checks                   ✓ one combining rule  ✓ real units  ✓ the cutoff has its tail  ✓ GPU styles exist
Notes                    ⚠ GAFF2 gave 1.836 g/cm³ for C6F14 (measured ~1.68): for fluorocarbons only with a refit
[Parameters...]  every type, charge and bonded term with its source       [Duplicate as a variant]
```

Typing runs on a worker thread. Its result (`TypedMolecule`) is cached per molecule, force field and QuantaBench
version, so a system types once and runs anywhere (section 9).

**3. Protocol.**

```
Template  [Liquid slab (surface tension) ▾]                                     Easy | Advanced
  1  Minimize            conjugate gradient
  2  Warm-up             NVT 10 ps at 0.5 fs
  3  Bulk                NPT 1 ns, 298.15 K, 1 atm, Nosé-Hoover
  4  Open the box        50 Å vacuum on each side
  5  Settle              NVT 2 ns
  6  Production          NVT 10 ns: pressure tensor every 10 fs, ρ(z) every 10 ps, a frame every 100 ps
Dispersion  ( ) Ewald for r⁻⁶ (CPU only)   (•) 16 Å cutoff + tail correction (GPU)   ( ) 12 Å + tail (screen)
            (the default: Brandon, 2026-11-22)
Restart     every 100 ps
Advanced    in.bulk / in.slab as written (read-only in v1; saved as a template of one's own in v2)
```

**4. Engine and resources.**

```
Engine    (•) LAMMPS 4 Jul 2026 (OPENMP, GPU/OpenCL)       ( ) OpenMM 8.6 (OpenCL; faster on GPUs, reduced precision)
Variant   [gpu-cutoff16: pair style on the GPU, PPPM on 6 threads ▾]   cpu-cutoff16, 4 threads
Run on    [This computer ▾]   21 ns/day (benchmark, verified) → 13 ns in 15 h
          (v2: Auto and the workers, each with its estimate and its time window)
Priority  below normal · threads within the machine's budget · time limit: none
Review    ✓ every style is in this build   ✓ GPU switches include device_type generic
          ✓ guard against a missing pair energy   ⓘ the Ewald option would run on the CPU only
[Write folder]   [Submit]
```

The review works like the build review (ADV-1): it never blocks, it explains, and Apply fixes a finding.

**5. Telemetry.**

```
Run 2 · C6F14 slab (OPLS-AA, cutoff16) · running on this computer · segment 2 (resumed on the GPU at 3.0 ns)
Step 7,412,000 / 13,000,000 · 21.0 ns/day · ETA 6 h 24 min · CPU 40 % · GPU 85 %
Plots   T | P | E | bulk ρ | ρ(z), latest | σ, running mean with block error | pull force (steered runs)
        raw / smoothed · export CSV
Alerts  none   (watched: QB_ERROR, T off target, density drift, lost atoms, the run stopped writing)
[Pause at the next restart]   [Stop now]   [Resume]   [Open folder]   [Log]
```

**6. Trajectory and analysis.**

```
Frames    slab.lammpstrj, 120 frames: the playback bar steps through them      [Open in OVITO]
Analyses  [✓] surface tension (Kirkwood-Buff; + LJ tail for cutoff variants)   → σ ± error, report
          [✓] density profile, tanh fit, plateau vs bulk, drift
          [ ] RDF  [ ] MSD / diffusion  [ ] PMF from repeated pulls (v2)
[Add to table]  σ_md, σ_md_err, ρ_md columns for this molecule in the Analyze table
```

"Add to table" is how MD feeds the QSAR work: the MD plan's `sigma_md` column with its own error bar and source flag.

## 5. Data model

Plain Python, no Qt, next to the existing writers (a new `md.sim` module), with JSON in and out like `project` and
`chem.recipes`. Every object has an `id` and records the QuantaBench version that wrote it.

```
System            what is simulated
  id, name, kind            "liquid-slab" | "bulk" | "surface" | (v2) "mixture" | "solution"
  components[]              molecule id (the project's), SMILES, count, role ("liquid", "adsorbate", "solvent", "ion")
  surface                   slab spec from build.surfaces (graphite/metal, layers, size), or none
  box                       lateral size, slab thickness, vacuum, or a full cell
  packing                   packmol | lattice, density and its source, tolerance, seed
  parameterization_id

Parameterization  the force field as applied to this system (ForceField in the backlog's words): a node tree
  id
  components[]              ComponentFF: component id, force field ("oplsaa" | "openff" | "gaff2" | "uff-rigid" |
                            surface preset | water model | ion set), resolution (all-atom | united-atom | coarse-
                            grained), charge method (OPLS, AM1-BCC by sqm, NAGL model, LigParGen, xTB/AIMNet2 for rigid
                            molecules), overrides, the TypedMolecule (today's to_dict: types, charges, LJ, bonded
                            terms, sources)
  physics                   vdw: LJ form, cutoff (the screening length), long range (none | bulk tail | slab tail in
                            the analysis | Ewald dispersion with its accuracy); coulomb: method and accuracy, slab
                            correction; combining rule and explicit cross terms; 1-4 scaling
  checks[]                  each compatibility check with its result; versions of the tools used

Protocol          an ordered list of stages, from a template
  id, template, name
  stages[]                  Stage: kind (minimize | nvt | npt | open-box | smd | anneal | production),
                            length (ps/ns or steps), timestep, T start/end, P, thermostat/barostat and damping,
                            groups and restraints, outputs (thermo, pressure tensor, ρ(z), frames, pull force)
  dispersion                ewald | cutoffN (+ tail correction in the analysis)
  restart_every

Run               one execution of a protocol on a system
  id, system_id, protocol_id, engine ("lammps" | "openmm"), variant (cpu-ewald, gpu-cutoff16, ...)
  resources                 machine, threads, GPU device, priority, time limit
  state                     written | queued | running | pausing | paused | done | failed | cancelled
  segments[]                Segment: machine, engine build (version, packages), variant, pid,
                            first and last step, restart file read and written, started, ended, exit status,
                            ns/day measured
  folder                    the run folder (inputs, outputs, restart files, analysis/)

Telemetry         read from the engine's own files, never a copy of them
  per segment               thermo series (step, time, T, P, E, density, box), pressure tensor, ρ(z),
                            pull force, ns/day from "Performance:" lines, the last time each file grew
  progress.json             written by the run supervisor every few seconds: step, ns done, ns/day, ETA, segment, pid

Trajectory        frames in the run folder
  files (LAMMPS dump; DCD/XTC from OpenMM), frame count and interval, the topology file they belong to

Analysis          a named analysis of a run
  kind (surface-tension | density-profile | rdf | msd | pmf | bulk-density), parameters, result JSON + Markdown
  in analysis/, and the columns it can add to the Analyze table
```

**Run folder.** It extends today's liquid-slab folder, so `md surface-tension` and the README commands keep working.

```
<run folder>/                 under the project's runs/ (or the output folder without a project)
    run.json                  the Run: system, parameterization and protocol (copies), engine, segments, state
    progress.json             the supervisor's heartbeat while it runs
    system.data  ff.params  in.bulk  in.slab  ...   the inputs, as today (README.txt, liquid_setup.json)
    log.*  pressure_tensor.txt  density_profile.txt  pull_force.txt     appended across segments
    *.restart.a/b             the engine's restart files; segment N+1 starts from the newest one
    frames/                   dumps (sparse by default)
    analysis/                 surface_tension.json/.md, ...
    STOP                      created to pause or stop: LAMMPS sees it (below) and ends its run cleanly
```

**Pausing and resuming LAMMPS.** Every input gets two lines:

```
variable exit equal is_file(STOP)
fix halt all halt 100 v_exit != 0 error soft
```

`fix halt` checks every 100 steps. With `error soft` it ends the current run and carries on with the input, which
then writes a restart file ([37](#sources)).

The supervisor writes the next segment's input from the protocol. That input:
- reads the newest restart file;
- skips the stages that ended before the restart step;
- re-creates the output fixes with `append` (`fix ave/time` has it since 17 Apr 2024 ([39](#sources)); pilot B's
  resume used it on `fix ave/chunk` too);
- runs the current stage to its end step with `run N upto` ([38](#sources)), then the stages after it.

This automates what pilot B's resume did by hand. LAMMPS reads a binary restart file only with the same executable
on the same platform, and fixes and computes must be given again ([50](#sources)). A run that moves to another
machine continues from a `write_data` file instead, which continues statistically similar rather than exactly.

**As built in SIM-1 PR 1** (docs/ENGINES.md, "MD runs as jobs"):
- `fix halt` checks every `pause_every` steps, the least common multiple of the output periods (10 ps with the
  defaults), so a pause never splits an averaging block. After every `run ... upto` the input checks whether the run
  stopped short of its end step and, if so, writes `<part>.pause.restart` and `<part>.pause.data` and quits with
  `QB_PAUSED <part> step N`: a halted run does not otherwise stop the input (with `error soft` the later runs are
  skipped and the input would write its final data file early).
- The periodic restart files stay `<part>.restart.a/b`, every 100 ps (was 1 ns); their step is read from the file's
  header.
- The supervisor picks the furthest checkpoint this machine and build may read (a binary restart only when the
  part's last segment ran here with the same LAMMPS build, else the pause data file), cuts the part's outputs at its
  step, and writes the segment's input from the protocol: the checkpoint read, the pair style, force field and
  k-space given again, finished stages skipped with `jump`, outputs appended.
- The rigid steered pull is a "rerun" part: a pause stops it and resume runs it again from the start.

**How it fits the project document (UI-2).** A `.qbproj` gains:

- `systems/`: one small JSON per System and Parameterization;
- `simulate` state in `project.json`: the open system and the stepper's state.

MD run folders go into the project's existing `runs/` beside batch run folders. A `kind` in `run.json` tells them
apart, so the Library dock and File > Open run folder... list both. Components refer to the molecules of
`molecules.jsonl` by id. MD results join the Analyze table by that id, the way Model... joins its predictions.

**How it fits the job model (UI-5).**
- The job dialog stays the molecular one. The Simulate stepper is MD's job dialog.
- A `Job` gains `kind = "md"`. Its progress is in ns, not molecules, and the Jobs dock shows ns/day and the ETA.
- An MD job does not run inside the GUI process. The queue starts the **run supervisor**
  (`quantabench-cli md run <folder>`) as its own process, which:
  - starts one engine process per segment through `chem.external.run_program` (log, process-tree cancel, run
    priority);
  - writes `progress.json`;
  - on Pause, creates the STOP file;
  - on Stop now, kills the engine after the STOP file has had its chance.
- Closing the window leaves the run going. Opening the project again re-attaches through `run.json` and
  `progress.json`, the way a worker's job is followed today.
- The queue's one-at-a-time rule applies per resource: one MD job per GPU, and MD threads within the machine's
  budget. Builds and MD can run side by side, as the pilots and CREST did.

**How it fits worker routing (v2).**
- The worker API gets a job of kind `md` that carries the written folder (a zip with a manifest; only known file
  names, the size checked). The worker runs it with its own engine build into its own output folder. The results
  come back as today: through the synced folder, else the archive.
- "Auto" picks the machine and variant with the shortest ETA from `md_throughput`, which `md estimate` already scales
  by atoms, and leaves out machines whose time window can't fit a segment.
- At the end of a window, the worker pauses with drain: it creates the STOP file. The run then waits for the window
  to reopen, or moves to another machine through a data file. Which of the two is a decision for Brandon (question
  6).
- Each finished segment's ns/day is recorded as a learned rate per machine, variant and size, next to the
  benchmark, as DIST-7 does for builds.

### Where MD runs live (question 2: SIM-1 PR 1 takes the recommendation, A)

Brandon asked how the conformer runs are set up today and whether both places are in SyncThing (2026-11-22).

**The facts:**
- **Batch runs** (the conformer and QSAR runs) go to the output folder in the settings. Brandon's is
  `~/SyncThing/QuantaBench/Runs`, which SyncThing syncs between the machines. Each run folder has one writer, the
  machine that runs it; the others only read it (ARCHITECTURE.md, "Batches are resumable"). The README asks for
  `jobs.sqlite-wal` and `jobs.sqlite-shm` in that SyncThing folder's `.stignore`, so SQLite's transient files never
  sync.
- **Projects** (`.qbproj`, UI-2) live wherever they are saved with File > Save as. None are in SyncThing yet. A
  project's `runs/` is synced only if the project itself is saved in a synced folder. Paths inside a project are
  relative, paths outside it are absolute, so a project can point at run folders elsewhere.
- **Sizes**, from the C6F14 pilots (9,780 atoms, 13 ns): 131-182 MB per run folder.
  - The pressure tensor every 10 fs: 48 MB.
  - The density profiles: 12 MB.
  - Frames every 100 ps: 52 MB (about 0.4 MB a frame).
  - Restart files: 2.9 MB each, two per stage. Data files: 2.7 MB each.

  The 17 liquids of Phase 1, two runs each, would be about 4-6 GB.

**The options:**

| | A: in the output folder (`Runs/`), the project points at them | B: in the project's `runs/` |
|---|---|---|
| Synced | Always: every machine sees the run | Only if the project is saved in SyncThing |
| Workers | A worker writes into its own output folder already, so local and remote runs land in the same place | A remote run would need copying into the project |
| Like today | Yes: batch runs work this way, and the Library dock lists the output folder | Like a project's builds |
| Moving the project | The links are absolute paths | Everything moves with the project |

**Recommendation: A**, the output folder, with the project pointing at the run, plus these lines in the SyncThing
folder's `.stignore`:

```
jobs.sqlite-wal
jobs.sqlite-shm
*.restart.a
*.restart.b
*.restart
STOP
frames
```

- Restart files are binary and readable only by the same LAMMPS on the same platform ([50](#sources)), so they are of
  no use to another machine.
- The STOP file is meant for the machine that runs the job only.
- Frames are the bulk of a run folder and are only needed to view the run. They come through the archive when
  another machine needs them. With frames written into `frames/` (section 5's run folder), the one pattern covers
  them; the pilots' `slab.lammpstrj` would need `*.lammpstrj`.
- The text outputs (pressure tensor, profiles, logs) stay synced, so σ can be analyzed on any machine.

**What SIM-1 PR 1 does** (2026-11-23): MD runs live in the output folder (A). The SyncThing folder's shared ignore
file (`.stglobalignore`, OPS-3, approved by Brandon) already keeps binary restart files (`*.restart`, `*.restart.a/b`,
`*.restart.[0-9]*`) and trajectory dumps (`*.lammpstrj`, `*.dcd`, `*.xtc`, ...) on the machine that wrote them, so
dumps stay named `slab.lammpstrj` rather than moving into `frames/`. `run.json`, `progress.json`, the logs, the text
outputs and the data files (`bulk_final.data`, `*.pause.data`, the portable checkpoints) sync. Not in the ignore file
yet: `STOP` and `supervisor.lock`. Both are small and harmless on another machine (a STOP only acts where the run
runs; a lock from another machine counts only while that machine's heartbeat in `progress.json` is fresh), but
adding `STOP` and `supervisor.lock` to `.stglobalignore` would keep them local, as this section proposed for STOP.

## 6. Splitting a calculation across machines

Brandon (2026-11-22): "We would need to be clever about what can be shared calculations or split, and we may need to
think about how that's done. We could do some different tests using Pilot A where we split calculations differently
so we know we get consistent results." **v1 stays on this computer; splitting across machines is v2, gated on that
test.**

There are three ways to split an MD calculation. A fourth, one time step spread over several machines with MPI, is
out of scope: the machines are linked by Tailscale over Wi-Fi or the internet, not a cluster network.

| | 1. Independent replicas | 2. One trajectory in segments, handed between machines | 3. GPU + CPU inside one machine |
|---|---|---|---|
| What | N runs of the same system from different seeds or packings, one per machine; σ is averaged | One run continued on the next machine at a segment boundary: `write_data` there, `read_data` and the protocol's remaining stages here | The pair style on the GPU, PPPM and bonded terms on OpenMP threads (`pair/only on`), as pilots B and C ran |
| Wall time | Shorter: the replicas run at the same time | Not shorter: the run is serial; it uses idle hours (the MacBook's night window) | Shorter: 21 against 3.9 ns/day in pilot B |
| Statistics | Independent samples: the error bar is the spread between replicas, which also catches slow fluctuations like pilot A's. Each replica pays its own settle time (2 ns here) | One correlated series, block-averaged as today. Equivalent to one run if the hand-off is clean | Those of one run |
| Portability | Each replica stays on one machine and its engine build; replicas from different builds are compared first | Needs the data file: binary restarts don't move between machines ([50](#sources)). Thermostat, barostat and averaging histories restart; both builds need the same styles | Nothing moves |
| Determinism | Not bitwise across machines (CPU double precision vs GPU mixed precision); statistically the same if the builds agree | Not exact: a data-file restart continues statistically similar. The join has to be checked, as pilot B's was | Mixed precision changed the fifth or sixth digit at pilot B's join; not bitwise reproducible, statistically the same |
| Good for | The 17-liquid sweep and error bars; Auto routing by ns/day | Long single runs that outlast one machine's window | Every cutoff run on a machine with a GPU |

**The consistency test** (pilot A; the plan is being written as `Runs/md-pilot-2026-11-20/split_test_plan.md` in
Brandon's output folder). What v2 needs from it:
- Replicas: σ from several shorter replicas, run on the workstation and the MacBook, against pilot A's single 10 ns,
  within the combined error bars.
- Hand-off: pilot A continued from a `write_data` file on the other machine. The energies and pressure at the join
  should agree to the precision of the two builds, and σ over the continued part should match the same length
  continued on the original machine, within the error bars.
- The GPU + CPU split can't be tested with pilot A (Ewald dispersion has no GPU style). Pilot B's checked CPU→GPU join
  covers it.

Machine splitting goes into the v2 worker work only for the strategies the test passes.

## 7. Scheduling by resource (SCHED-3)

Brandon (2026-11-22): "Maybe we can include a job queue or scheduler? If one type of job is better suited for the CPU
and another for a GPU, then we can tell the user that both jobs can run efficiently at the same time."

Brandon (2026-11-23): "The point of the scheduler is to compare what each job needs, then optimize and make one job go
faster by dedicating just GPU, CPU, or both." So the scheduler's first aim is **the fastest device assignment for a
given job**: it compares what each job needs (from the run timings, MON-1's samples and the MD benchmark's ns/day per
variant) and gives a job the GPU only, the CPU only, or both, whichever finishes it soonest (for GPU MD: the GPU plus
the 4-6 threads its PPPM needs, below). Running jobs side by side is the second aim: it does so when that helps, and
says so.

**A resource profile per job.** Before a job starts, the queue estimates what it needs:
- CPU threads, and how busy they really are;
- the GPU and its memory;
- RAM;
- disk I/O (CREST through WSL wrote about 20 MB/s before CREST-SCRATCH-1).

The estimate comes from what QuantaBench already measures:
- the run timings (stage times, CPU time, the run's own CPU share);
- MON-1's machine samples (GPU utilization and VRAM);
- MD-BENCH-1's ns/day per variant;
- the method (AIMNet2 batched on the GPU, xTB and CREST on the CPU).

**GPU MD still needs CPU threads.** The LAMMPS timing breakdowns of the pilots' GPU runs:

| run | threads | PPPM (Kspace) | bonded | pair (the wait for the GPU) | CPU use |
|---|---|---|---|---|---|
| B, 16 Å, GPU segment | 6 | 48 % | 27 % | 9 % | 237 % of 600 % |
| C, 12 Å | 4 | 58 % | 28 % | 4 % | 173 % of 400 % |

PPPM and the bonded terms on the CPU took 75-86 % of the loop time, so a GPU MD job's speed depends on its CPU
threads. Its profile is "the GPU + 4 to 6 threads", not "the GPU".

**How the queue uses the profiles:**
- **A ledger per machine:** threads within the budget (free threads, Max CPU), one GPU-heavy job per GPU, RAM
  headroom, one heavy-disk job at a time.
- **A job starts when its profile fits.** On the 12-core workstation, a GPU MD job with 4-6 threads leaves the rest for
  CPU lanes; the pilots ran that way beside CREST.
- **The user is told:** "C6F14 slab (GPU + 6 threads) and the CREST batch (6 lanes x 1 thread) can run side by side.
  The slowdown is not measured yet." Once runs have been timed alone and together, it gives the measured slowdown.
  The same rule could start the next queued job early when its profile fits beside the running one.
- **Across machines**, the profile feeds Auto: a CPU-heavy job goes where the cores are free, a GPU job where the GPU
  is.

Today the GUI queue runs one build at a time, lanes split the CPU within a build, and MD jobs would run beside it as
their own processes (section 5). SCHED-3 replaces the one-at-a-time rule with the ledger.

## 8. Engines

| | LAMMPS (first) | OpenMM (v2) |
|---|---|---|
| For | Everything QuantaBench writes today: liquids and slabs, surfaces, rigid and (later) flexible steered MD, polymers; any pair style, including Ewald dispersion ([48](#sources)) | ML-potential MD through OpenMM-ML (AIMNet2, MACE, ANI and others in its list ([29](#sources))); mixed ML/MM with `createMixedSystem` ([29](#sources)); OpenFF systems without the LAMMPS export's caveats ([24](#sources)); later free energies (Open Free Energy) |
| How it runs | The `lmp` executable as a process; the Python module only for short checks and tests, as today | A Python process in an environment with OpenMM (and torch for ML) |
| Restart | Binary restart files, same executable and platform only; `write_data` for moves ([50](#sources)) | Checkpoints only on the same hardware and OpenMM version; `saveState` XML moves between platforms and versions ([12](#sources)) |
| Status here | Built, benchmarked, pilots done | Not started (ROADMAP 0.4, FORK-9) |

**GPU support:**

| | AMD (the workstation: RX 9060 XT, Windows) | Apple (the MacBook: M3 Max) | NVIDIA |
|---|---|---|---|
| LAMMPS GPU package | OpenCL: the Windows installer's `lmp.exe` has the GPU package in OpenCL mode, mixed precision, for AMD, Intel and NVIDIA ([36](#sources)). Needs `device_type generic` here (lesson 1). HIP builds need ROCm ([34](#sources)) | OpenCL "already included with the OS" on many Macs ([34](#sources)), with an `applegpu` device type ([35](#sources)). No build here yet: the MacBook runs conda-forge LAMMPS on OpenMP (FORK-8) | CUDA ("only tested on Linux and unsupported on Windows") or OpenCL ([34](#sources)) |
| LAMMPS KOKKOS | HIP, built from source; the Windows installer has only serial and OpenMP KOKKOS ([36](#sources)) | No Apple GPU backend listed ([33](#sources)) | CUDA ([33](#sources)) |
| OpenMM | OpenCL in the base pip package; HIP with `openmm[hip6]` / `[hip7]` ([11](#sources)), with Windows and Linux wheels ([32](#sources)) | OpenCL, supplied by macOS ([11](#sources)); a Metal plugin in beta, built from source ([31](#sources)) (FORK-9) | CUDA with `openmm[cuda12]` ([11](#sources)) |
| OpenMM-ML | The model runs in PyTorch: AMD's ROCm torch on Windows (`.venv-rocm`). Untested here | PyTorch on MPS. Untested here | CUDA torch |

Notes:
- **LAMMPS on Windows:** use the installer's `lmp.exe`. The PyPI wheel failed 5-10 % of identical runs there
  (MD-1).
- **LAMMPS-GUI's bundles** use the same OpenCL GPU package and do no MPI runs through the GUI ([4](#sources)).
- **OpenMM-ML's AIMNet2** wraps AIMNet2's calculator in a Python callback that runs at every step, so its speed is
  limited by that path ([30](#sources)). On a ROCm GPU the stock calculator would meet the Warp problem QuantaBench
  solved in GPU-1, unless `aimnet` has changed since. OpenMM-ML's `ase` option takes any ASE calculator
  ([29](#sources)), which is a possible route for QuantaBench's Warp-free one. Both have to be measured. Expect
  small systems: a PFAS molecule with a few hundred waters, not a 10,000-atom slab.

## 9. Force fields

| Route | Covers | Charges | Where it runs | What we know |
|---|---|---|---|---|
| OPLS-AA + Watkins-Jorgensen (`oplsaa`) | Alkanes, perfluoroalkanes, (fluoro)benzenes | OPLS-AA's own | In process, any OS | Tsige & Grest's route ([46](#sources)). C6F14 in the pilots: bulk density 1.664-1.672 g/cm³ (measured about 1.67-1.68); σ 10.7-11.7 mN/m, with 16 Å + tail reproducing Tsige & Grest's 11.25 within 4 % |
| OpenFF Sage 2.3.0 (`openff`) | Organic molecules in Sage's coverage | AM1-BCC by AmberTools' sqm; **NAGL** in v2: graph-network charges, independent of the conformer, production model `openff-gnn-am1bcc-1.0.0` ([25](#sources), [26](#sources)) | The md conda env (Linux, macOS; WSL on Windows) through `md/ff_driver.py` | Exported with Interchange's LAMMPS writer. Interchange's LAMMPS input is a single point to be edited, and LAMMPS has no switching function, so energies differ from OpenMM/GROMACS ([24](#sources)). QuantaBench reads only the parameters and writes its own inputs. The C6F14 pilot folder is written, not run |
| GAFF2 + AM1-BCC (`gaff2`) | Organic molecules | AM1-BCC by sqm | The md conda env | 1.836 against about 1.67-1.68 g/cm³ for C6F14 (lesson 7). For fluorocarbons, only with the Träg & Zahn refit ([47](#sources)) through `--overrides`, once its numbers are taken from the paper |
| UFF, rigid (`md.lammps`) | Any molecule; no bonded terms | xTB or AIMNet2 charges from the pipeline | Any OS | Steered pull in vacuum; a gas-phase lower bound on the work |
| Surfaces | Graphite (AMBER parm94, Steele 1973 in `md.params`), metal slabs | None (not polarizable) | Any | Sage is a small-molecule force field, so a surface always needs its own parameters and an explicit cross-term rule, recorded with sources. Ions: Joung-Cheatham (in `md.params`) |

Rules the workspace enforces:
- **Coverage before writing.** A molecule the chosen force field cannot type is refused with the reason, and the
  routes that can type it are offered.
- **One force field per system,** except surfaces, ions and water, whose parameters are listed separately with their
  sources. Mixing two molecular force fields in one box is out of scope.
- **Typed once, run anywhere.** OpenFF and GAFF2 typing needs AmberTools and OpenFF from conda-forge in a Linux or
  macOS environment (the MD plan), so on the workstation it means WSL or typing on the MacBook. The parameterization
  is stored with the system, so the run itself needs only `lmp`.
- **Defaults per class come from benchmarks** (section 9.1): OPLS-AA for perfluoroalkanes, OpenFF for the other
  classes until their benchmarks exist, GAFF2 only with a refit.
- **LigParGen** (OPLS-AA charges for more groups) is allowed (Brandon, 2026-11-22). It is a web service that receives
  the SMILES, so the review says so, and the charges it returns are stored with their source. How it is called (a
  file uploaded by hand, or a programmatic interface if it has one) and its terms of use are checked when it is built.

### 9.1 The parameterization builder: force fields as composable nodes

Brandon (2026-11-22): "make the force field selection and how coarse each one is kind of like how COMSOL lets you drag
and drop various physics into a setup. This would take a lot of trial and erroring, but I think it would be a
tremendous value add."

The Force field step becomes a tree of nodes, the way a COMSOL model is a tree of physics interfaces. Nodes are added
from a palette (drag and drop or a menu), and each node has its own settings and a status.

```
Parameterization: PFOA on graphite in water                                        [+ Add node ▾]
├─ Components
│  ├─ PFOA⁻ x 1            [OpenFF Sage 2.3.0 ▾]  [all-atom ▾]  charges [NAGL ▾]      ✓ typed
│  ├─ Graphite, 3 layers   [graphite LJ, AMBER parm94 ▾]  [all-atom, frozen ▾]  no charges
│  ├─ Water x 3,000        [TIP3P ▾]  [all-atom, rigid ▾]
│  └─ Na⁺ x 1              [Joung-Cheatham (TIP3P set) ▾]
├─ Physics
│  ├─ Van der Waals        LJ 12-6 · cutoff (screening length) [16 Å] · long range [tail correction ▾]
│  ├─ Electrostatics       PPPM 1e-5 · slab correction [auto]
│  └─ Combining rule       [Lorentz-Berthelot ▾] · cross terms graphite-water [table...]
└─ Checks
   ✓ one combining rule and one 1-4 scaling for the molecules
   ✓ real units everywhere   ✓ net charge 0
   ⚠ cutoff without a long-range treatment would bias σ low → tail correction on
```

**Component nodes**, one per molecule type, surface slab, water or ion set, each with:
- its force field;
- its resolution:
  - all-atom (OPLS-AA, GAFF2, OpenFF);
  - united-atom (a united-atom force field would need its own typing, which QuantaBench doesn't have);
  - coarse-grained (bead-spring models such as the one CHARMM-GUI's Polymer Builder equilibrates with before going back
    to all-atom ([9](#sources)));
- its charges and overrides.

The data model carries the resolution from the start; v1 and v2 offer all-atom only.

**Physics nodes**, global to the system:
- **Van der Waals:** the LJ cutoff, which Brandon calls the "screening length" and wants as an option when building
  force fields. Its long-range treatment:
  - none;
  - the analytic tail for a homogeneous bulk ([40](#sources));
  - the slab tail added in the analysis (PR #37);
  - Ewald dispersion with its accuracy.

  The tail correction is switched on automatically for a cutoff run.

  MD-LR (2026-11-23) built this node's options and defaults as `md.long_range`: the two methods, the cutoff labelled
  "LJ cutoff (screening length)", the Coulomb and dispersion k-space settings (slab-safe by default), and
  `recommend_long_range(...)`, which picks them from the setup (slab or bulk, box, size, whether LAMMPS can use a GPU,
  screen or reference) with a reason and a speed. The node shows the recommendation, lets the user override any
  value, and shows `check_settings(...)`'s warnings and `differences(...)` beside it (docs/ENGINES.md).
- **Electrostatics:** PPPM and its accuracy, the slab correction where the box has one non-periodic direction.
- **Combining rule and cross terms:** the mixing rule, and explicit pair coefficients where two parts disagree (a
  surface against a liquid).
- **1-4 scaling:** taken from the molecular force field.

**Compatibility checks**, run on every change:
- **Combining rules.** OPLS-AA mixes geometrically; GAFF2 and OpenFF use Lorentz-Berthelot (`md.forcefield`). LAMMPS
  mixes all unlisted pairs with one rule, so a mixed system either chooses one rule or gets explicit cross terms, and
  the check says which pairs are affected.
- **1-4 scaling.** OPLS-AA scales 1-4 pairs by 0.5/0.5, GAFF2 and OpenFF by 0.5/0.8333 (`md.forcefield`). LAMMPS sets
  1-4 scaling once for the system with `special_bonds` (exceptions per pair style need a hybrid pair style), so two
  molecular force fields with different 1-4 scaling don't share a box. The check refuses that until such handling
  exists.
- **Units:** everything in LAMMPS `real` units, converted once where a backend uses other units (Interchange,
  ParmEd).
- **The cutoff:**
  - a warning for a cutoff without a tail correction (lesson 4);
  - an error for LAMMPS' bulk tail correction in a slab ([40](#sources));
  - a note that Ewald dispersion has no GPU style (lesson 3).
- **Water models and ion sets** are listed with the force fields they were fitted with; another pairing gets a
  warning.

**Benchmark-backed defaults per class.** A class gets a default only from a benchmark against measurements:

| Class | Default | Evidence |
|---|---|---|
| Perfluoroalkanes | OPLS-AA + Watkins-Jorgensen, all-atom, 16 Å + tail, GPU | Pilots A-C; Tsige & Grest ([46](#sources)) |
| Other PFAS classes (ethers, alcohols, acids, sulfonyl, aromatics, Cl/Br) | none yet: OpenFF Sage offered, with a "not benchmarked" mark | Pending: the MD plan's Phase 1 (17 liquids), and a literature review being written (`Literature Reviews/2026-11-22 MD surface tension cutoffs and force fields/` in Brandon's SyncThing folder) |

**Trial and error made cheap.** "Duplicate as a variant" copies a system and changes one node. Pilots A, B and C were
such a set: the same system, three dispersion treatments. In v2 a variant set runs as a sweep, and its results come
back as one comparison table, like a COMSOL parametric study.

**Build order:**
- **v1 (fits PR 2):** the tree with one component node (the liquid, or the rigid adsorbate with its slab), the physics
  nodes for the cutoff and its long-range treatment and for electrostatics, the combining rule shown read-only, and
  the checks.
- **v2:** several components (water, ions, a surface with a liquid), explicit cross-term tables, NAGL and LigParGen
  charges, variants run as sweeps.
- **v3:** united-atom and coarse-grained resolutions.

## 10. Protocol templates

A template is a named list of stages with defaults, like a recipe (`chem.recipes`). The user's own templates are
saved in the settings, and every template writes the same kind of folder.

| Template | Stages | Outputs and checks | Analysis |
|---|---|---|---|
| Minimize | conjugate gradient | final structure, the largest force | energy before and after |
| NVT | velocities, Nosé-Hoover (Langevin for rigid bodies and single ions, as `md.lammps` does) | thermo | T drift, energy |
| NPT | NVT warm-up, then NPT (isotropic, or anisotropic for solids); LJ tail correction allowed here, in a homogeneous bulk ([40](#sources)) | box, density every ps | bulk density ± error, equilibration time |
| Liquid slab (σ) | MD-2: minimize, warm-up, bulk NPT, open the box, settle, production; 16 Å + tail by default (Brandon, 2026-11-22) | pressure tensor every 10 fs, ρ(z) every 10 ps, sparse frames; plateau vs bulk density, drift, evaporation | σ by Kirkwood-Buff ± block error; LJ tail for cutoff variants; density profile and its fit |
| Steered MD + Jarzynski | equilibrate, then N pulls from independent starts (seeds or frames), constant-velocity spring | pull force and distance per pull | work per pull; PMF by Jarzynski's exponential average ([49](#sources)) and the cumulant form, with a bootstrap error |
| Annealing | heat to T_high, hold, cool at a set rate, repeat; `run ... start/stop` keeps a ramp going across segments ([38](#sources)) | T, density | density against T |

Notes:
- **v1 has the first four templates and the existing single rigid pull.** The repeated pulls, flexible molecules and
  solvent come in v2.
- **LAMMPS' `fix smd` is marked unmaintained.** Its documentation recommends `fix colvars` or `fix plumed` instead
  ([41](#sources)). `fix colvars` gives harmonic restraints, umbrella sampling, ABF and metadynamics, with PMFs as
  output ([42](#sources)). The flexible steered MD of v2 should be written on Colvars, after checking that each
  machine's build has the COLVARS package (question 9).
- **Annealing** is how CHARMM-GUI equilibrates polymer melts ([9](#sources)), and a Forcite task
  ([13](#sources)). It matters for the sorbent work later.
- **The setups these templates make** (a liquid slab, a bulk liquid, a solvation box, a pull on a surface, a polymer
  melt, a liquid-liquid interface), with their inputs, defaults and measurements, and the pre-flight checks that go
  with them, are section 15.

## 11. Monitoring and analysis

| Quantity | From | Live | After the run |
|---|---|---|---|
| T, P, E, density, box | thermo lines in the log | plots with raw/smoothed and CSV export (as LAMMPS-GUI ([2](#sources))) | averages, drift |
| σ | pressure tensor file | running mean with its block error | `md.surface_tension`: σ ± error, LJ tail, report |
| Density profile ρ(z) | `fix ave/chunk` file | the latest profile | tanh fit, plateau vs bulk, vapour density, drift |
| Pull force, work | `pull_force.txt` | force against distance | work per pull; PMF from repeated pulls (v2) |
| RDF (v2) | LAMMPS `compute rdf` during the run, or MDAnalysis' `rdf` from the frames ([44](#sources)) | — | g(r), coordination numbers |
| MSD and diffusion (v2) | LAMMPS `compute msd`, or MDAnalysis' `msd` ([44](#sources)) | MSD so far | D from the linear part, with its fit range shown |
| Speed | "Performance:" lines, step counts per segment | ns/day, ETA | the learned rate for this machine and variant |

Design points:
- **Telemetry comes from the engine's files.** QuantaBench tails them on a worker thread and never asks the engine.
- **Alerts** come from the run folder:
  - QB_ERROR in the log;
  - a temperature off target;
  - the density drifting;
  - a file that stopped growing while the process lives.
  The machine's CPU, GPU and temperatures come from MON-1's monitor.
- **Analyses that need frames are optional dependencies.** MDAnalysis has been LGPL since 2.8 ([45](#sources)). Its
  `LinearDensity` works only for orthorhombic cells of fixed volume ([43](#sources)), so the density profile stays
  QuantaBench's own, from LAMMPS' output.
- **OVITO stays the external viewer for big trajectories.** Its viewport embeds as a PySide6 widget
  ([23](#sources)). Several of its analysis modifiers are Pro-only ([22](#sources)), so QuantaBench does not depend
  on them.

## 12. What to build, in which order

### SIM-1 v1: two PRs

**PR 1, MD runs as jobs** (no new panel). **Built, 2026-11-23** (branch `feat/sim-1-md-run-jobs`); what differs
from the list below:
- `md.sim` has the System, Parameterization, Protocol (parts of stages), Run and Segment dataclasses; System and
  Parameterization carry what today's writers know (the full parameters stay in `liquid_setup.json`).
- **The default dispersion is not changed here.** MD-LR (#52) made the long-range method and the LJ cutoff user
  options with a recommendation from the setup (`md.recommend_long_range`): `md liquid-slab` without a choice already
  writes 16 Å + tail where LAMMPS can use a GPU and converged pppm/disp on a CPU-only machine.
- Beyond the list: MD jobs in the GUI's job queue (a kind-"md" job starts the supervisor as a process of its own, runs
  beside builds, one MD job at a time on this computer; the Jobs dock shows ns, ns/day and the ETA; Pause and Resume;
  Simulate > Run MD folder...), and `md run` on folders written before SIM-1. The job dialog stays the molecular one;
  MD on workers is v2.
- Tests run a 540-atom hexane slab with the `lmp` executable (skipped without one): a pause and a resume give one
  continuous pressure-tensor and density-profile series with the same energy at the join; a stopped segment resumes
  from its last restart file; a run that moved to another build resumes from the data file; a later segment uses
  other engine settings. The CPU-then-GPU segment test runs where `QB_TEST_MD_GPU=1` is set and LAMMPS has the GPU
  package; it passed on the workstation's RX 9060 XT (2026-11-23).

The plan as written:
- `md.sim`: System, Parameterization, Protocol/Stage, Run/Segment as JSON dataclasses, filled by today's writers.
  `write_liquid_slab` and `write_smd_setup` gain `run.json`, the STOP-file lines and segment-ready outputs.
- The run supervisor and its commands:
  - `quantabench-cli md run DIR [--variant V] [--threads N]`, which runs a folder's stages one segment at a time at
    the run priority and writes `progress.json`;
  - `md status DIR`;
  - `md pause DIR`, through the STOP file;
  - `md resume DIR`, which reads the newest restart file, appends outputs and runs `upto`.
- A telemetry reader: incremental parsing of the log, pressure tensor, density profile and pull-force files, and
  ns/day per segment, so a resumed run's speed is known.
- The liquid slab's default dispersion becomes `cutoff16` (in `md liquid-slab` too; today it is `ewald`), with the
  tail correction in the report (Brandon, 2026-11-22).
- Tests with tiny LAMMPS runs (they skip without LAMMPS):
  - pause and resume give one continuous pressure-tensor series;
  - a killed segment resumes from its last restart file;
  - a CPU segment can be followed by a GPU segment.

**PR 2, the Simulate panel:**
- The stepper (section 4) for two kinds of system: the pure liquid slab and the molecule on a surface (today's
  surface builder and rigid pull).
- The parameterization builder's v1 (section 9.1): one component node with `oplsaa`, `openff` or `gaff2`, coverage and
  availability; the physics nodes for the cutoff and electrostatics; the checks.
- Templates: minimize, NVT, NPT, liquid slab, and the rigid pull.
- LAMMPS variants from `md.bench.can_run` with the estimate from `md_throughput`.
- **Run on: this computer only.**
- Submit puts a kind-`md` job into the queue and starts the supervisor.
- Telemetry plots (pyqtgraph), Pause, Resume, Stop. After the run: the σ report, frames on the playback bar, Open in
  OVITO, Add to table.
- Re-attaching to running runs when a project opens.
- A scripted smoke test under xvfb with screenshots.

**Not in v1:** workers for MD and splitting across machines, OpenMM, mixtures and solutions, repeated pulls, RDF/MSD,
editable inputs.

### v2

- **MD on the workers:** the `md` job kind in the worker API, Auto by ETA, time windows through segments, learned
  rates. Splitting by replicas or hand-offs (section 6) only for the strategies the pilot A consistency test passes.
- **Scheduling by resource (SCHED-3, section 7):** resource profiles and the per-machine ledger, with GPU and CPU jobs
  side by side.
- **Sweeps:** one protocol over many molecules, for the 17 test liquids of the MD plan, with results in the Analyze
  table.
- **Systems:** mixtures and solutions with Packmol, a molecule on graphite in water, counter-ions (MD-1's lessons).
- **Steered MD:** flexible, on Colvars, with N repeated pulls and the Jarzynski PMF; annealing.
- **Force fields:** the builder's multi-component version (section 9.1); NAGL charges; the Träg & Zahn GAFF2
  overrides; more OPLS-AA types through LigParGen (allowed).
- **OpenMM:** classical MD with OpenFF through Interchange's `to_openmm` ([24](#sources)), on the same Run and
  Segment model.
- **Analysis:** RDF, MSD and bulk-density analyses; re-evaluating saved frames with other settings (`rerun`, lesson
  10); Advanced input editing saved as one's own template.

### v3

- **OpenMM-ML:** AIMNet2 (or AceFF, MACE-OFF) MD and ML/MM for a PFAS molecule with its first water shells or on a
  surface.
- **Free energies:** hydration free energies through Open Free Energy, for a validation subset.
- **Builders:** amorphous polymer and sorbent cells (Amorphous Cell- and RadonPy-like), with annealing; united-atom
  and coarse-grained resolutions in the parameterization builder.
- **Views and exports:** a trajectory player synced with the telemetry plots (Tier 2 in UI_PLAN.md), GROMACS export,
  a Mac GPU route (FORK-8 or FORK-9).

## 13. Risks

- **The force field dominates the error.**
  - GAFF2's density is 10 % high for C6F14.
  - OPLS-AA covers few PFAS classes.
  - The other classes rely on charges nobody has tested for σ (the MD plan's main risk).

  A polished workspace does not fix that; the review must keep saying which route is validated for which class.
- **Runs take days on shared machines.**
  - CREST and MD compete for the same cores.
  - A laptop sleeps.
  - A GUI process should not own a multi-day run, hence the separate supervisor and the re-attach.
- **LAMMPS builds differ:** the installer, the PyPI wheel, conda-forge; packages present or not (COLVARS, KOKKOS
  untested); GPU tuning by vendor (lesson 1). The Windows wheel is non-deterministic (MD-1).
- **Restart files are not portable.**
  - LAMMPS restart files are "only expected to be read correctly by the same LAMMPS executable on the same
    platform" ([50](#sources)).
  - Fixes and computes must be re-specified on restart, and some fixes "will not restart exactly"
    ([50](#sources)).
  - OpenMM's checkpoints are tied to the hardware and the OpenMM version ([12](#sources)).

  Moving a run between machines goes through a data file (or OpenMM's state XML), which continues statistically
  similar rather than exactly.
- **The fast route needs a correction, the exact one is slow.** GPU variants need a cutoff, so σ depends on the tail
  correction and on the slab's lower density (lesson 4). The Ewald route is CPU-only, 2-3 times slower and, in pilot
  A, noisier, with its mesh still to check (lesson 10).
- **`fix smd` is unmaintained** ([41](#sources)): the steered-MD path has to move to Colvars or PLUMED.
- **Disk and sync:**
  - The pressure tensor every 10 fs is 10⁶ lines per 10 ns.
  - Frames add up quickly.
  - Run folders in a synced folder move all of that between machines.

  Frames should be sparse by default, and big files may need to stay out of SyncThing (question 3).
- **OpenMM-ML speed and GPU support** on AMD and Apple are unmeasured (section 8).
- **Splitting can change results.** Replicas from different engine builds and hand-offs through data files are
  statistically, not exactly, the same run (section 6). Hence the consistency test before v2 uses them.
- **The composable builder invites combinations nobody has validated.** The checks catch the mechanical conflicts
  (combining rules, 1-4 scaling, units). Whether a combination is physically sound comes only from benchmarks, so
  defaults are offered only where a benchmark exists.
- **Scope.** The Materials Studio-class builders are years of work; v1 deliberately wraps what exists.

## 14. Open questions for Brandon

1. **v1 scope:** the two PRs above, this computer only, the liquid slab and the rigid pull? Or should v1 already send
   MD jobs to the workers?
   **Answered (Brandon, 2026-11-22):** be clever about what can be shared or split, and test it with pilot A. v1 stays
   on this computer; splitting across machines is v2, gated on the consistency test (section 6).
2. **Where MD runs live:** in the project's `runs/`, or in the output folder `Runs/` with the project pointing at
   them? The facts, the two options and a recommendation (the output folder, with `.stignore` lines
   for restarts, the STOP file and frames) are in section 5, "Where MD runs live". **SIM-1 PR 1 takes the
   recommendation** (the output folder; the approved `.stglobalignore` keeps restarts and dumps local); still to
   confirm, and whether `STOP` and `supervisor.lock` join the ignore file.
3. **Frames and big files:** the default frame interval, and should frames and restart files stay out of SyncThing
   (an ignore pattern) and move only through the archive? **Answered in part:** the approved `.stglobalignore` keeps
   restart files and dumps out of SyncThing. PR 1 keeps a frame every 100 ps (the pilots') and moves restart files from
   every 1 ns to every 100 ps (local files, 2.9 MB each, two per part).
4. **The default force field per class once pilots A and B are written up:** OPLS-AA for perfluoroalkanes, OpenFF
   Sage for the rest, GAFF2 only with the Träg & Zahn refit?
   **Answered in part (Brandon, 2026-11-22):** force-field selection becomes a composable builder, like physics in
   COMSOL, with the resolution per component and the screening length as an option (section 9.1). The defaults per
   class come from benchmarks; the literature review is pending, and the other classes need Phase 1.
5. **LigParGen:** may OPLS-AA charges for more PFAS groups come from the web service (the SMILES leave the machine),
   or local routes only? **Answered (Brandon, 2026-11-22): allowed.**
6. **A run that meets the end of a machine's time window:** wait for the window to reopen on the same machine (an
   exact continuation), or move to a free machine through a data file (statistically similar)? **PR 1 (this
   computer only) waits:** a pause keeps both checkpoints, and a resume on the same machine and build continues from
   the binary one; a resume elsewhere would use the data file. The choice for workers waits for the split test.
7. **The σ default:** 16 Å on the GPU with the tail correction (pilot B: reproduced Tsige & Grest, 21 ns/day, slab 2 %
   below the bulk density), or Ewald dispersion on the CPU (pilot A: no correction, 7.6 ns/day, its mesh still to
   check)? And the C6F14 target: Freire 2006 (12.33), which OPLS-AA misses by 5-13 %, or Caço 2003 (11.36)?
   **Answered (Brandon, 2026-11-22):** 16 Å + tail on the GPU is the default ("that default is fine"). The target
   stays Freire 2006 (12.33) with no offsets; the OPLS-AA shortfall is documented, not corrected.
8. **OpenMM's place:** in v2 next to the workers, or after the workers and sweeps? Is AIMNet2 MD (a PFAS molecule
   in water or on graphite) worth a v2 spike? (Open; PR 1's Run and Segment model has an `engine` field for it.)
   **Answered (Brandon, 2026-11-23):** keep both, LAMMPS and OpenMM: "we can just add a note as a part of the
   selection process (MD Engine: OpenMM, LAMMPS etc) and explain that OpenMM runs much faster at reduced accuracy."
   OPENMM-1 adds OpenMM as a second engine for liquid slabs now (the same folder, files and supervisor; the engine is
   chosen when the folder is written or before it starts, with that note). LAMMPS stays the reference engine;
   OpenMM-ML (AIMNet2 MD) is still v2.
9. **Steered MD:** move to Colvars (needs the COLVARS package in every machine's LAMMPS), or keep `fix smd` for the
   rigid pull and use Colvars only for the flexible one? **PR 1 keeps `fix smd` for the rigid pull** (section 10's
   recommendation), as a "rerun" part.
10. **Sharing machines by default:** may an MD job start while a CREST or QSAR batch runs on the same machine, and at
    which priority and thread share?
    **Answered (Brandon, 2026-11-22):** a scheduler that knows which jobs suit the CPU and which the GPU, and tells the
    user when both can run efficiently at once (section 7, SCHED-3). The default thread shares come from the measured
    profiles.
11. **The 17-liquid Phase 1:** run it from the CLI with the v1 supervisor as soon as PR 1 lands, before the panel?
    (Possible once PR 1 is merged: `md liquid-slab` per liquid, then `md run --detach` one at a time.)
12. **The Mac's GPU:** try a LAMMPS OpenCL build (FORK-8) or OpenMM Metal (FORK-9) first, or neither for now?
    (Open; PR 1 runs the Mac's conda-forge LAMMPS on the CPU.)
    **Answered (Brandon, 2026-11-23):** OpenMM on the Apple GPU ("I approve writing the code for Apple GPU for
    OpenMM"). It runs through Apple's OpenCL in single precision (no FP64), and σ is computed afterwards in double
    precision (OPENMM-1). The LAMMPS OpenCL build (FORK-8) is deferred: without FP64 it could only be single precision.

## 15. Setup templates and pre-flight checks

Brandon (2026-11-23): "For the MD, I would like to eventually have some default geometries and common set ups." A
**setup template** is a system kind (the geometry), a protocol template (section 10) and the analyses that turn the
run into numbers, with defaults that a benchmark backs where one exists. A **pre-flight check** runs before a folder
is written (and again before and during `md run`) and catches the usual reasons an MD setup fails or gives wrong
numbers. They are BACKLOG SIM-2 (templates) and SIM-3 (checks); both build on SIM-1's run folders, so every template
runs, pauses and resumes the same way.

### 15.1 Setup templates

| Template | Geometry and inputs | Defaults | Stages | What it measures | Status |
|---|---|---|---|---|---|
| Liquid slab (surface tension) | One liquid; lateral edge, slab thickness or molecules, vacuum | 40 × 40 Å lateral (50 × 50 "full"), 100 Å slab, 50 Å vacuum on each side, 298.15 K, 1 fs; long range from `recommend_long_range` (16 Å + tail on a GPU, converged pppm/disp on a CPU) | Minimize, NVT warm-up, bulk NPT 1 ns, open the box, NVT 2 ns settle + 10 ns production | σ by Kirkwood-Buff ± block error, with the LJ tail for a cutoff; plateau vs bulk density; vapour density; interface width; drift | Built (MD-2, MD-LR, SIM-1) |
| Bulk liquid (density, heat of vaporization) | One liquid in a cube; plus the same molecule alone in the gas phase | At least 500 molecules and a box over 2 rc + 2 Å; 12 Å + LAMMPS' homogeneous tail (`recommend_long_range("bulk")`, valid in a uniform liquid ([40](#sources))); 298.15 K, 1 atm; NPT 1 ns + 5 ns production; the gas-phase molecule with Langevin dynamics and no cutoff | Bulk: minimize, warm-up, NPT, NPT production. Gas: NVT | Density ± error; ΔHvap = ⟨U_pot⟩gas − ⟨U_pot⟩liquid / N + RT; optionally the thermal expansion and compressibility from the volume and enthalpy fluctuations | The bulk stages exist (`in.bulk`); the gas phase and the analysis to build |
| Solvation box | A solute (any charge) in a solvent box | 12 Å of solvent around the solute (and more than rc + 2 Å); TIP3P water with Joung-Cheatham ions (`md.params`), counter-ions to neutralize, optional salt; NPT 1 ns + 10 ns production | Minimize (solute restrained), NVT, NPT, production | Solute-solvent RDFs and coordination numbers, the first-shell water count, the solute's conformers; the start for hydration free energies (v3) | To build (v2: mixtures and solutions) |
| Rigid pull or adsorption on a surface | An adsorbate on a slab (graphite layers, a metal), its height; vacuum (v1) or water (v2) | Graphite, 2-3 frozen layers; 3.5 Å start height; spring 10 kcal/mol/Å²; 1e-4 Å/fs over 15 Å; v2: 10 pulls from independent starts | Soft start, NVT 50 ps, the pull (`fix smd`; v2 Colvars) | Work and peak force per pull; v2: the Jarzynski PMF with a bootstrap error, the adsorption energy at contact | The rigid pull is built (0.2; a "rerun" part in SIM-1); flexible and solvated pulls v2 |
| Polymer melt | N chains of a repeat unit: degree of polymerization, tacticity, end groups | About 20 chains of 40 repeat units (at least ~5,000 atoms); packed at a low density and compressed; annealed from well above Tg (CHARMM-GUI's melts: Tm + 50 K to 298 K ([9](#sources))) | Minimize, compression NPT, annealing cycles (section 10's template), NPT production 10 ns | Density; Tg from density against T on cooling; radius of gyration and end-to-end distance; cohesive energy density and the solubility parameter | v3 (builders; a typing route for polymers) |
| Liquid-liquid interface | Two immiscible liquids stacked along z, periodic, no vacuum | 40 × 40 Å, 50 Å of each liquid; NPT with the barostat on z only (fixed area); one combining rule for both liquids or explicit cross terms (section 9.1) | Minimize, NVT warm-up, z-only NPT, production | Interfacial tension by Kirkwood-Buff (two interfaces, no vacuum); each component's density profile; interface width; mutual solubility; molecular orientation at the interface | v2 (mixtures) |

Rules for every template:
- **The same folder.** `run.json` with its parts and stages, inputs with the STOP lines, a README with every
  parameter and its source, so `md run/pause/resume`, the Jobs dock and the analyses work for all of them.
- **Easy and Advanced** (QwikMD's split, section 3.2): a template asks for a few inputs (the molecules, the size, the
  temperature, the purpose); every other value has a default with its reason, shown under Advanced.
- **No unbenchmarked default.** A template's defaults per force-field class come from benchmarks (section 9.1); where
  none exists the review says so.
- **Validation per template** (SIM-2's acceptance): the bulk liquid against the measured density and ΔHvap of hexane
  and C6F14 and against the force field's own published values; the liquid slab against pilots A-C; the solvation box
  against the water model's published density.

### 15.2 Pre-flight checks

| Check | What goes wrong | How it is checked | When | Severity and fix |
|---|---|---|---|---|
| Missing or guessed parameters, atom typing of unusual groups | A type, bond, angle or torsion without a parameter, or one filled by analogy (a generic SMIRKS, GAFF's parmchk2 placeholders, a zero torsion), most often for unusual groups (perfluoroalkyl ethers, sulfonyl, carboxylates). OPLS-AA typing covers only alkanes, perfluoroalkanes and (fluoro)benzenes (lesson 8) | Every parameter carries its source (`TypedMolecule`); a source that says analogy, generic or default is listed with its atoms; coverage per force field before anything is written | Before writing | Error for a missing parameter; warning for a guessed one, with the routes that type the group (section 9) |
| Charges and net charge | A non-integer total; a liquid whose molecules are not neutral; charges from one method with another force field; a symmetry-broken AM1-BCC set | The sum per molecule equals its formal charge within 1e-4 e; counter-ions for a charged solute; the charge method named with the force field; symmetry-equivalent atoms within 0.01 e | Before writing | Error for the net charge, warnings otherwise; fixes: neutralize with ions, average equivalent atoms |
| Cutoff against half the box | The minimum image breaks (a pair sees two images); LAMMPS warns "Bond/angle/dihedral extent > half of periodic box length" | rc + skin below half the smallest periodic width; a slab's lateral edge over 2 rc + 1 Å (`plan_box` today); under NPT also at the smallest box the barostat may reach | Before writing; again from the first NPT averages | Error |
| Overlaps after packing | Atoms of different molecules too close: exploding first steps, "Lost atoms"; rings speared by chains in polymers | The closest intermolecular contact (`min_contact`; refused below 0.8 Å today); the largest force and the energy after the minimization; a ring-spearing test for polymers (Amorphous Cell has one ([14](#sources))) | After packing; after the minimization | Error below 0.8 Å, warning below the packing tolerance; fix: pack again with another seed or at a lower density |
| k-space accuracy and mesh for slabs | LAMMPS' error estimate assumes a uniform density, so a slab's vacuum gets it too coarse a mesh (pilot A: 2 x 2 x 8, σ 1.36 mN/m low) | MD-LR's spacing rule and `check_settings` (G·rc, h·G, Coulomb accuracy); the grids LAMMPS printed are read back from the first segment's log and compared with the rule; the slab correction for `p p f` boxes | Before writing; after the first segment's setup | Warning (MD-LR's); error when the log's grid is coarser than the rule |
| Time step against constraints | 2 fs with unconstrained X-H bonds or flexible water; a rigid body or a light atom at too long a step | The fastest bond's period from masses and force constants (at least ~10 steps per period); the constraints present (`fix shake`, rigid water, rigid bodies) | Before writing | Error for 2 fs with free X-H bonds; warning otherwise; fix: 1 fs, or SHAKE |
| Vacuum and slab thickness | Images interact across too little vacuum; a slab too thin has no bulk-like plateau; a small lateral box shifts σ (capillary waves) | Vacuum at least max(3 rc, 40 Å) on each side; the slab at least twice the interface width plus ~30 Å of plateau; the lateral edge at least 40 Å for σ | Before writing; after the run (the plateau check `md.surface_tension` already makes) | Warning |
| Thermostat and barostat damping | Too small: wild fluctuations; too large: slow equilibration; weak-coupling schemes give the wrong fluctuations | Tdamp about 100 time steps and Pdamp about 1000 (LAMMPS' advice for Nosé-Hoover ([51](#sources))); Langevin for a single rigid body (`md.lammps`); an isotropic barostat for a liquid, z-only at fixed area for a liquid-liquid interface, none for a liquid-vapour slab | Before writing | Warning; fix: the defaults |
| Equilibration length and drift | Production that starts before the density, the energy or the slab has settled; a drifting slab | Bulk: the density's halves within their errors and its slope; slab: the plateau against the bulk density and the slab's drift (both in `md.surface_tension`); the equilibrated start found automatically (Chodera's method ([52](#sources))) as the skip time | After each part, from its outputs; live in the Jobs dock and the panel | Warning; fix: a longer equilibration or a later skip time |
| Error bars from block averages | An error bar from correlated samples is too small; pilot A's blocking curve never levelled off | Block averages with at least 4 blocks longer than the correlation time, the statistical inefficiency and the Flyvbjerg-Petersen blocking curve (in `md.surface_tension`); a warning when the curve has not levelled off; independent replicas for an honest error (section 6) | After the run | Warning; fix: a longer production or replicas |

The checks section 9.1 already lists (one combining rule and one 1-4 scaling, units, a cutoff without its tail) and the
engine review of section 4 (every style in this build, the GPU switches, the pair-energy guard) join them, with two
more: the disk space of the outputs (the pressure tensor every 10 fs is 48 MB per 10 ns) and the run time against the
machine's time window (`md estimate`).

**How they run** (SIM-3): `md.preflight`, pure Python where it can be, with `check(system, parameterization,
protocol) -> [Finding(id, severity, message, fix)]`. The writers call it before writing and refuse on an error
(`--force` for experts); the supervisor runs the after-the-fact checks on a part's outputs before the next part starts
and puts them in `progress.json`; the Simulate panel shows them like the build review (ADV-1): it explains, Apply
fixes a finding, and only an error blocks. Every finding is stored in `run.json` (`parameterization.checks`), so a
result carries what was checked.

## Sources

Internal, not public: the pilot reports (`Runs/md-pilot-2026-11-20/pilot_A_report.md`, `pilot_B_report.md`,
`pilot_C_report.md`, `pilots_summary.md`), the planned split test (`split_test_plan.md` there, being written) and the
literature review on cutoffs and force fields (pending), all in Brandon's synced folders.

Fetched 2026-11-22. Vendor pages of commercial tools list features without detail; the table says only what they
say.

1. LAMMPS-GUI overview (editor, runs LAMMPS in a thread, Charts, Slide Show, variables, tutorials): https://lammps-gui.lammps.org/overview.html
2. LAMMPS-GUI, monitoring output (Output window, Charts with smoothing and export, Variables window): https://lammps-gui.lammps.org/output.html
3. LAMMPS-GUI basic usage (Run, Stop finishes the current step, progress bar): https://lammps-gui.lammps.org/basic_usage.html
4. LAMMPS-GUI installation (bundles with the GPU package in OpenCL mixed precision; no MPI through the GUI): https://lammps-gui.lammps.org/installation.html
5. QwikMD plugin page (CHARMM36, Easy and Advanced runs, SMD, Live View, analyses, `.qwikmd` and `.infoMD`): https://www.ks.uiuc.edu/Research/vmd/plugins/qwikmd/
6. QwikMD documentation (live simulations through NAMD's interactive MD; analysis tab): https://www.ks.uiuc.edu/Research/vmd/plugins/qwikmd/documentation.html
7. QwikMD project page (Ribeiro et al., Sci. Rep. 2016, 6, 26536): https://www.tcbg.illinois.edu/Research/qwikmd/
8. CHARMM-GUI Multicomponent Assembler (Nat. Commun.; abstract): https://par.nsf.gov/biblio/10552842
9. CHARMM-GUI Polymer Builder (CGenFF, melts and solutions, engines, annealing protocol): https://par.nsf.gov/servlets/purl/10296171
10. OpenMM-Setup (README): https://github.com/openmm/openmm-setup
11. OpenMM user guide, getting started (platforms; pip extras for CUDA and HIP; OpenCL on macOS): https://docs.openmm.org/latest/userguide/application/01_getting_started.html
12. OpenMM user guide, running simulations (platform names incl. HIP and Metal; checkpoints vs saveState): https://docs.openmm.org/development/userguide/application/02_running_sims.html
13. BIOVIA Materials Studio Forcite Plus datasheet: https://www.3ds.com/fileadmin/PRODUCTS-SERVICES/BIOVIA/PDF/biovia-material-studio-forcite-plus.pdf
14. BIOVIA Materials Studio Amorphous Cell datasheet: https://www.3ds.com/fileadmin/PRODUCTS-SERVICES/BIOVIA/PDF/biovia-material-studio-amorphous-cell.pdf
15. Scienomics MAPS platform: https://www.scienomics.com/maps-platform/
16. LAMMPS pre- and post-processing tools (MAPS, moltemplate, Packmol, OVITO, LAMMPS-GUI, others): https://www.lammps.org/ecosystem/prepost/
17. SAMSON (Wikipedia: OneAngstrom, extensions, Python API, GROMACS in the cloud): https://en.wikipedia.org/wiki/SAMSON
18. SAMSON GROMACS Wizard (blog): https://blog.samson-connect.net/simplify-molecular-dynamics-setup-with-the-gromacs-wizard-in-samson
19. SAMSON OpenMM Wizard (extension page): https://samson-connect.net/extensions/b951ba8f-2f5e-d114-a635-f1c41af40fb7
20. Packmol.jl documentation (Packmol's new implementation): https://m3g.github.io/Packmol.jl/stable/
21. LAMMPS Howto moltemplate: https://docs.lammps.org/Howto_moltemplate.html
22. OVITO modifiers reference (Pro-only ones marked): https://ovito.org/manual/reference/pipelines/modifiers/index.html
23. OVITO Python `ovito.gui` (`create_qwidget`, PySide6): https://www.ovito.org/manual/python/modules/ovito_gui.html
24. OpenFF Interchange, exporting to other software (OpenMM, GROMACS, LAMMPS, Amber; LAMMPS caveats): https://docs.openforcefield.org/projects/interchange/en/stable/using/output.html
25. OpenFF NAGL models (PyPI; `openff-gnn-am1bcc-1.0.0` production): https://pypi.org/project/openff-nagl-models
26. OpenFF toolkit `NAGLToolkitWrapper` (charges from the molecular graph): https://docs.openforcefield.org/projects/toolkit/en/latest/api/generated/openff.toolkit.utils.toolkits.NAGLToolkitWrapper.html
27. RadonPy (README): https://github.com/radonpy/radonpy
28. CHAPERONg (bioRxiv 2023): https://www.biorxiv.org/content/10.1101/2023.07.01.546945.full.pdf
29. OpenMM-ML user guide (potentials, `createMixedSystem`, `ase`): https://openmm.github.io/openmm-ml/dev/userguide.html
30. AIMNet2 with OpenMM (openmm-ml's AIMNet2 integration, PythonForce): https://isayevlab.github.io/aimnetcentral/external/openmm/
31. openmm-metal, beta Metal platform for OpenMM: https://github.com/bdenhollander/openmm-metal
32. OpenMM-HIP-6 on PyPI (Windows and Linux wheels): https://pypi.org/project/OpenMM-HIP-6/8.3.0/
33. LAMMPS accelerator packages: https://docs.lammps.org/Speed_packages.html
34. LAMMPS GPU package (CUDA, OpenCL, HIP; precision; pair/only; macOS OpenCL): https://docs.lammps.org/Speed_gpu.html
35. LAMMPS `package` command (`device_type`, `pair/only`): https://docs.lammps.org/package.html
36. LAMMPS Windows packages (GPU package in OpenCL mode, KOKKOS serial/OpenMP only, LAMMPS-GUI installers): https://packages.lammps.org/windows.html
37. LAMMPS `fix halt` (`error soft`; stop when a file exists): https://docs.lammps.org/fix_halt.html
38. LAMMPS `run` (`upto`, `start`/`stop`): https://docs.lammps.org/run.html
39. LAMMPS `fix ave/time` (`append`, since 17 Apr 2024): https://docs.lammps.org/fix_ave_time.html
40. LAMMPS `pair_modify` (`tail yes` only for homogeneous bulk liquids, not slabs): https://docs.lammps.org/pair_modify.html
41. LAMMPS `fix smd` (Jarzynski-based PMF; unmaintained, use colvars or plumed): https://docs.lammps.org/fix_smd.html
42. LAMMPS `fix colvars`: https://docs.lammps.org/fix_colvars.html
43. MDAnalysis `LinearDensity` (orthorhombic, fixed volume): https://docs.mdanalysis.org/dev/documentation_pages/analysis/lineardensity.html
44. MDAnalysis analysis modules (rdf, msd, lineardensity): https://docs.mdanalysis.org/stable/documentation_pages/analysis_modules.html
45. MDAnalysis 2.8.0 release (relicensed to LGPL): https://mdanalysis.org/2024/11/22/release-2.8.0
46. Tsige, M.; Grest, G. S. Surface tension and surface orientation of perfluorinated alkanes. J. Phys. Chem. C 2008, 112, 5029: https://doi.org/10.1021/jp710678w
47. Träg, J.; Zahn, D. Improved GAFF2 parameters for fluorinated alkanes and mixed hydro- and fluorocarbons. J. Mol. Model. 2019, 25, 39: https://doi.org/10.1007/s00894-018-3911-5
48. in 't Veld, P. J.; Ismail, A. E.; Grest, G. S. Application of Ewald summations to long-range dispersion forces. J. Chem. Phys. 2007, 127, 144711: https://doi.org/10.1063/1.2770730
49. Jarzynski, C. Nonequilibrium equality for free energy differences. Phys. Rev. Lett. 1997, 78, 2690 (cited by [41](#sources)): https://doi.org/10.1103/PhysRevLett.78.2690
50. LAMMPS `read_restart` (binary restart files: same executable and platform; fixes re-specified): https://docs.lammps.org/read_restart.html
51. LAMMPS `fix nh` (Tdamp "around 100 timesteps", Pdamp "around 1000 timesteps"): https://docs.lammps.org/fix_nh.html
52. Chodera, J. D. A simple method for automated equilibration detection in molecular simulations. J. Chem. Theory Comput. 2016, 12, 1799: https://pmc.ncbi.nlm.nih.gov/articles/PMC4945107
