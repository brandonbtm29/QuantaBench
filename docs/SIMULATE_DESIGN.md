# Simulate workspace design (RES-1)

Status: **proposal, 2026-11-22. Waits for Brandon's approval** (BACKLOG DEC-1); SIM-1 starts after that. No code
here. Sources are numbered at the end ([Sources](#sources)); repository facts point at the module or file.

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
  comes in v2 for ML-potential MD (OpenMM-ML) and the OpenFF-native path.
- **Force fields keep a source for every parameter**, and coverage is checked before a run. OPLS-AA stays the
  reference for perfluoroalkanes; OpenFF Sage (with NAGL charges later) is the general route; GAFF2 needs a
  fluorocarbon refit before it is used for PFAS liquids.
- **SIM-1 v1 is two PRs:**
  1. MD runs as jobs: a run supervisor with segments, resume and live telemetry, plus the CLI.
  2. The Simulate panel for the two setups that exist today, the liquid slab and the rigid steered pull, run on
     this computer.

  Workers, OpenMM, mixtures and the extra analyses come in v2 and v3.

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
| `cli_md` | `md smd`, `md parse`, `md liquid-slab`, `md surface-tension`, `md estimate` | |

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

Pilot runs: `Runs/md-pilot-2026-11-20` in Brandon's output folder (pilot C's report; first analyses of pilots A and
B written 2026-11-22, their reports still in progress). Each lesson below comes with what the design does about it.

| # | Lesson | Consequence for the design |
|---|---|---|
| 1 | On the AMD RX 9060 XT (OpenCL), LAMMPS' default "AMD_GPU" tuning built empty neighbor lists: no nonbonded forces, no error. `device_type generic` fixes it (`md.liquid.gpu_switches`). | An engine *variant* carries its switches. Every input keeps the pair-energy guard. A new GPU, driver or build gets the `run 0` CPU/GPU check (`md.bench`) before a production run. |
| 2 | The fastest setup was `-sf hybrid gpu omp -pk gpu 1 device_type generic pair/only on -pk omp N`: 15.2 against 10.2 ns/day with everything on the GPU (9,780 atoms). | The variant is chosen from measured ns/day, not guessed. Real runs report their speed back as a learned rate. |
| 3 | `lj/long/coul/long` and `pppm/disp` have no GPU styles. | The protocol's dispersion choice limits the engine variants. The review says why the GPU is greyed out. |
| 4 | A plain 12 Å LJ cutoff read σ = 7.96 ± 0.32 mN/m for C6F14, about 30 % low (Tsige & Grest: 11.25 ([46](#sources))). The tail correction added 3.16, giving 11.12. The slab liquid was 4.7 % less dense than the bulk NPT. LAMMPS' own tail correction assumes a homogeneous bulk liquid and must not be used for a slab ([40](#sources)). First analyses: Ewald dispersion (pilot A) 10.70 ± 0.70; 16 Å (pilot B) 9.60 raw, 11.71 with the tail. | Cutoff variants always report raw σ, σ with the tail, and the plateau-vs-bulk density check. Ewald is the reference. The template offers "16 Å + tail on the GPU" as the fast route only with that label. |
| 5 | Long runs need restart, resume and appended outputs. Pilot B moved from CPU to GPU at 3 ns through a hand-written resume input (`read_restart`, outputs cut at the restart step and appended, `run ... upto`). The MacBook's GAFF2 slab stopped at its window's deadline. The analysis could not read pilot B's speed, because its log was split. | Segments are first-class (section 5). Resume is a button. Outputs append. Telemetry and speed are kept per segment. |
| 6 | The MacBook runs conda-forge LAMMPS (22 Jul 2025) with OpenMP, with restart files every 100 ps, `caffeinate` and a deadline script. | Each machine finds its own engine build. The restart interval follows the machine's window. A Mac GPU build of LAMMPS is FORK-8. |
| 7 | GAFF2 + AM1-BCC gave too high a density for C6F14: 1.83 against about 1.68 g/cm³ (MacBook bulk NPT, 2026-11-21). | Force-field choice per class, with a warning on fluorocarbons. Refits (Träg & Zahn ([47](#sources))) go through the existing LJ overrides. |
| 8 | OPLS-AA typing covers only alkanes, perfluoroalkanes and (fluoro)benzenes. | Coverage is checked before anything is written, with the alternatives offered. |
| 9 | The pilots ran beside the paused CREST rebuild at below-normal priority on a thread budget. | MD jobs use the same priority and budget settings as builds. A GPU is a resource with its own lane. |

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

- **Web services that upload structures** (CHARMM-GUI, LigParGen) as a required step: unpublished PFAS stay local
  unless Brandon decides otherwise per molecule.
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
|▣ |     run 2  ● 7.4/12 ns|                                                      |   OPLS-AA (W&J)    |
|▣ | ▸ C6F14 slab (OpenFF) |                                                      | 3 Protocol      ✓  |
|  | ▸ PFOA on graphite    |                                                      |   Liquid slab (σ)  |
|  | [+ New system]        |  |◀ ◀  ▶  ▶▶|  frame 74 / 120        ns 7.4        | 4 Engine        ✓  |
|  |                       |                                                      |   LAMMPS cpu-ewald |
|  |                       |                                                      | 5 Telemetry     ●  |
|  |                       |                                                      | 6 Analysis         |
|  +-----------------------+------------------------------------------------------+--------------------+
|  | Telemetry: [T] [P] [E] [ρ] [σ running] [pull force]   7.4 / 12 ns · 7.6 ns/day · ETA 14 h · CPU 66 % |
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

**2. Force field.**

```
Force field  (•) OPLS-AA + Watkins-Jorgensen 2001   ( ) OpenFF Sage 2.3.0 + AM1-BCC   ( ) GAFF2 + AM1-BCC
             ( ) UFF, rigid molecule (steered pull only)                              ⓘ which to choose
Coverage     ✓ all 20 atoms typed                    (OpenFF, GAFF2: installed in the md environment ✓)
Charges      fixed OPLS-AA charges        (OpenFF/GAFF2: AM1-BCC by sqm; NAGL in v2)
Mixing       geometric · 1-4 scaling 0.5 / 0.5       Overrides: none [Load...]
Notes        ⚠ GAFF2 gave 1.83 g/cm³ for C6F14 (measured ~1.68): use it for fluorocarbons only with a refit
[Parameters...]  every type, charge and bonded term with its source (the setup JSON)
```

Typing runs on a worker thread. Its result (`TypedMolecule`) is cached per molecule, force field and QuantaBench
version, so a system types once and runs anywhere (section 7).

**3. Protocol.**

```
Template  [Liquid slab (surface tension) ▾]                                     Easy | Advanced
  1  Minimize            conjugate gradient
  2  Warm-up             NVT 10 ps at 0.5 fs
  3  Bulk                NPT 1 ns, 298.15 K, 1 atm, Nosé-Hoover
  4  Open the box        50 Å vacuum on each side
  5  Settle              NVT 2 ns
  6  Production          NVT 10 ns: pressure tensor every 10 fs, ρ(z) every 10 ps, a frame every 100 ps
Dispersion  (•) Ewald for r⁻⁶ (CPU only)   ( ) 16 Å cutoff + tail correction (GPU)   ( ) 12 Å + tail (screen)
Restart     every 100 ps
Advanced    in.bulk / in.slab as written (read-only in v1; saved as a template of one's own in v2)
```

**4. Engine and resources.**

```
Engine    (•) LAMMPS 4 Jul 2026 (OPENMP, GPU/OpenCL)       ( ) OpenMM (v2)
Variant   [cpu-ewald, 8 threads ▾]   gpu-cutoff16: pair style on the GPU, PPPM on 4 threads (needs 16 Å)
Run on    [This computer ▾]   7.6 ns/day (benchmark, verified) → 13 ns in 1 d 17 h
          (v2: Auto and the workers, each with its estimate and its time window)
Priority  below normal · threads within the machine's budget · time limit: none
Review    ✓ every style is in this build   ✓ GPU switches include device_type generic
          ✓ guard against a missing pair energy   ⚠ ewald: no GPU variant
[Write folder]   [Submit]
```

The review works like the build review (ADV-1): it never blocks, it explains, and Apply fixes a finding.

**5. Telemetry.**

```
Run 2 · C6F14 slab (OPLS-AA, ewald) · running on this computer · segment 2 (resumed at 3.0 ns)
Step 7,412,000 / 13,000,000 · 7.4 ns/day · ETA 18 h · CPU 66 % · GPU —
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

Parameterization  the force field as applied to this system (ForceField in the backlog's words)
  id, backend               "oplsaa" | "openff" | "gaff2" | "uff-rigid"
  per component             TypedMolecule (today's to_dict: types, charges, LJ, bonded terms, sources, overrides)
  surface terms             LJ preset for the slab (md.params) and the mixing rule for the cross terms
  charges                   method (OPLS, AM1-BCC by sqm, NAGL model, xTB/AIMNet2 for rigid molecules)
  mixing, special_lj/coul, coverage report, versions of the tools used

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

## 6. Engines

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

## 7. Force fields

| Route | Covers | Charges | Where it runs | What we know |
|---|---|---|---|---|
| OPLS-AA + Watkins-Jorgensen (`oplsaa`) | Alkanes, perfluoroalkanes, (fluoro)benzenes | OPLS-AA's own | In process, any OS | Tsige & Grest's route ([46](#sources)). Bulk density of C6F14 1.664-1.672 g/cm³ in the pilots (measured about 1.67-1.68) |
| OpenFF Sage 2.3.0 (`openff`) | Organic molecules in Sage's coverage | AM1-BCC by AmberTools' sqm; **NAGL** in v2: graph-network charges, independent of the conformer, production model `openff-gnn-am1bcc-1.0.0` ([25](#sources), [26](#sources)) | The md conda env (Linux, macOS; WSL on Windows) through `md/ff_driver.py` | Exported with Interchange's LAMMPS writer. Interchange's LAMMPS input is a single point to be edited, and LAMMPS has no switching function, so energies differ from OpenMM/GROMACS ([24](#sources)). QuantaBench reads only the parameters and writes its own inputs. The C6F14 pilot folder is written, not run |
| GAFF2 + AM1-BCC (`gaff2`) | Organic molecules | AM1-BCC by sqm | The md conda env | 1.83 against about 1.68 g/cm³ for C6F14 (lesson 7). For fluorocarbons, only with the Träg & Zahn refit ([47](#sources)) through `--overrides`, once its numbers are taken from the paper |
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
- **The default per class waits for the pilots** (question 4): OPLS-AA for perfluoroalkanes, OpenFF for the other
  classes, GAFF2 only with a refit.
- **LigParGen** (OPLS-AA charges for more groups) is a web service that sends the SMILES out; it needs Brandon's
  decision (question 5).

## 8. Protocol templates

A template is a named list of stages with defaults, like a recipe (`chem.recipes`). The user's own templates are
saved in the settings, and every template writes the same kind of folder.

| Template | Stages | Outputs and checks | Analysis |
|---|---|---|---|
| Minimize | conjugate gradient | final structure, the largest force | energy before and after |
| NVT | velocities, Nosé-Hoover (Langevin for rigid bodies and single ions, as `md.lammps` does) | thermo | T drift, energy |
| NPT | NVT warm-up, then NPT (isotropic, or anisotropic for solids); LJ tail correction allowed here, in a homogeneous bulk ([40](#sources)) | box, density every ps | bulk density ± error, equilibration time |
| Liquid slab (σ) | MD-2: minimize, warm-up, bulk NPT, open the box, settle, production | pressure tensor every 10 fs, ρ(z) every 10 ps, sparse frames; plateau vs bulk density, drift, evaporation | σ by Kirkwood-Buff ± block error; LJ tail for cutoff variants; density profile and its fit |
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

## 9. Monitoring and analysis

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

## 10. What to build, in which order

### SIM-1 v1: two PRs

**PR 1, MD runs as jobs** (no new panel):
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
- Tests with tiny LAMMPS runs (they skip without LAMMPS):
  - pause and resume give one continuous pressure-tensor series;
  - a killed segment resumes from its last restart file;
  - a CPU segment can be followed by a GPU segment.

**PR 2, the Simulate panel:**
- The stepper (section 4) for two kinds of system: the pure liquid slab and the molecule on a surface (today's
  surface builder and rigid pull).
- Force fields `oplsaa`, `openff`, `gaff2` with coverage and availability.
- Templates: minimize, NVT, NPT, liquid slab, and the rigid pull.
- LAMMPS variants from `md.bench.can_run` with the estimate from `md_throughput`.
- **Run on: this computer only.**
- Submit puts a kind-`md` job into the queue and starts the supervisor.
- Telemetry plots (pyqtgraph), Pause, Resume, Stop. After the run: the σ report, frames on the playback bar, Open in
  OVITO, Add to table.
- Re-attaching to running runs when a project opens.
- A scripted smoke test under xvfb with screenshots.

**Not in v1:** workers for MD, OpenMM, mixtures and solutions, repeated pulls, RDF/MSD, editable inputs.

### v2

- **MD on the workers:** the `md` job kind in the worker API, Auto by ETA, time windows through segments, learned
  rates.
- **Sweeps:** one protocol over many molecules, for the 17 test liquids of the MD plan, with results in the Analyze
  table.
- **Systems:** mixtures and solutions with Packmol, a molecule on graphite in water, counter-ions (MD-1's lessons).
- **Steered MD:** flexible, on Colvars, with N repeated pulls and the Jarzynski PMF; annealing.
- **Force fields:** NAGL charges; the Träg & Zahn GAFF2 overrides; more OPLS-AA types if Brandon allows LigParGen.
- **OpenMM:** classical MD with OpenFF through Interchange's `to_openmm` ([24](#sources)), on the same Run and
  Segment model.
- **Analysis:** RDF, MSD and bulk-density analyses; Advanced input editing saved as one's own template.

### v3

- **OpenMM-ML:** AIMNet2 (or AceFF, MACE-OFF) MD and ML/MM for a PFAS molecule with its first water shells or on a
  surface.
- **Free energies:** hydration free energies through Open Free Energy, for a validation subset.
- **Builders:** amorphous polymer and sorbent cells (Amorphous Cell- and RadonPy-like), with annealing.
- **Views and exports:** a trajectory player synced with the telemetry plots (Tier 2 in UI_PLAN.md), GROMACS export,
  a Mac GPU route (FORK-8 or FORK-9).

## 11. Risks

- **The force field dominates the error.**
  - GAFF2's density is 9 % high for C6F14.
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
- **The fast route is biased.** GPU variants need a cutoff, so σ depends on the tail correction and on the slab's
  lower density (lesson 4).
- **`fix smd` is unmaintained** ([41](#sources)): the steered-MD path has to move to Colvars or PLUMED.
- **Disk and sync:**
  - The pressure tensor every 10 fs is 10⁶ lines per 10 ns.
  - Frames add up quickly.
  - Run folders in a synced folder move all of that between machines.

  Frames should be sparse by default, and big files may need to stay out of SyncThing (question 3).
- **OpenMM-ML speed and GPU support** on AMD and Apple are unmeasured (section 6).
- **Scope.** The Materials Studio-class builders are years of work; v1 deliberately wraps what exists.

## 12. Open questions for Brandon

1. **v1 scope:** the two PRs above, this computer only, the liquid slab and the rigid pull? Or should v1 already send
   MD jobs to the workers?
2. **Where MD runs live:** in the project's `runs/`, like batch runs, or always in the output folder `Runs/` with the
   project pointing at them?
3. **Frames and big files:** the default frame interval, and should frames and restart files stay out of SyncThing
   (an ignore pattern) and move only through the archive?
4. **The default force field per class once pilots A and B are written up:** OPLS-AA for perfluoroalkanes, OpenFF
   Sage for the rest, GAFF2 only with the Träg & Zahn refit?
5. **LigParGen:** may OPLS-AA charges for more PFAS groups come from the web service (the SMILES leave the machine),
   or local routes only?
6. **A run that meets the end of a machine's time window:** wait for the window to reopen on the same machine (an
   exact continuation), or move to a free machine through a data file (statistically similar)?
7. **The σ default:** Ewald dispersion on the CPU (the reference, about 7 ns/day here), or 16 Å on the GPU with the
   tail correction (faster, biased by the slab density)? And the C6F14 reference: Freire 2006 (12.33) or Caço 2003
   (11.36)?
8. **OpenMM's place:** in v2 next to the workers, or after the workers and sweeps? Is AIMNet2 MD (a PFAS molecule
   in water or on graphite) worth a v2 spike?
9. **Steered MD:** move to Colvars (needs the COLVARS package in every machine's LAMMPS), or keep `fix smd` for the
   rigid pull and use Colvars only for the flexible one?
10. **Sharing machines by default:** may an MD job start while a CREST or QSAR batch runs on the same machine, and at
    which priority and thread share?
11. **The 17-liquid Phase 1:** run it from the CLI with the v1 supervisor as soon as PR 1 lands, before the panel?
12. **The Mac's GPU:** try a LAMMPS OpenCL build (FORK-8) or OpenMM Metal (FORK-9) first, or neither for now?

## Sources

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
