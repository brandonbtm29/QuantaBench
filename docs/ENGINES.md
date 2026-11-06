# Engines

QuantaBench computes energies and properties with several engines. The in-process ones (MMFF, dxtb,
AIMNet2, tblite) are Python packages. The external ones (xtb, CREST, MOPAC, Psi4) are separate
programs that run as subprocesses: their OpenMP and BLAS stay out of PyTorch's way, a crash cannot take
the GUI down, a run can be cancelled by killing the process tree, and Linux-only programs can run
through WSL on Windows. `quantabench-cli doctor` lists what it found and where.

| Engine key | What | Family | Runs | Solvent | Windows | Notes |
|---|---|---|---|---|---|---|
| `mmff` | MMFF94s / UFF (RDKit) | force field | in process | none | yes | conformer pre-optimization |
| `gfn1-xtb` | GFN1-xTB (dxtb) | tight binding | in process, CPU/GPU | none | yes | batched on GPU |
| `gfn2-xtb` | GFN2-xTB (dxtb + libcint) | tight binding | in process | none | dxtb fork | PyPI dxtb: Linux/WSL with `tad-libcint<0.4`; fork: all OSes (below) |
| `aimnet2` | AIMNet2 (ωB97M-D3 ML) | ML potential | in process, CPU/GPU (CUDA: stock aimnet; ROCm, MPS: Warp-free `aimnet_torch`) | none | yes | weights download |
| `tblite-gfn2`, `tblite-gfn1` | GFN-xTB (tblite) | tight binding | in process | ALPB, GBSA, CPCM | yes (fork wheel) | GFN2 without libcint |
| `xtb-gfn2`, `xtb-gfn1`, `gfnff` | the xtb program | tight binding / force field | external | ALPB, GBSA | yes | native optimizer |
| `mopac-pm7`, `mopac-pm6` | PM7, PM6-D3H4X (MOPAC) | semi-empirical | external | COSMO | yes | energy = ΔHf |
| CREST | iMTD-GC conformer search | conformers | external | ALPB | via WSL | `--conformers crest` |
| Psi4 | DFT/HF QM stage | DFT | external (own env) | PCM | yes | `--qm wb97x-d/def2-svp` |

`method="auto"` tries AIMNet2 > GFN2-xTB (dxtb) > GFN2-xTB (tblite) > GFN2-xTB (xtb) > GFN1-xTB > MMFF,
and with a solvent prefers the engines that can model it. When it passes an installed engine over for a
molecule's elements, the notes say so ("auto: AIMNet2 (ML, wB97M-D3) has no parameters for Sn; using
GFN2-xTB (dxtb + libcint)"), so a batch that mixes levels of theory is visible in `summary.csv`.

## Elements, salts, metals, charge and spin

| Engine | Elements |
|---|---|
| AIMNet2 | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| GFN1/GFN2-xTB (dxtb, tblite, xtb), GFN-FF | H to Rn (Z = 1-86) |
| MMFF94s | organic elements and common ions; **UFF** takes over for the rest (Sn, Hg, hypervalent iodine, B, radicals), noted in the results and in the `forcefield` column |

What the standardization does with research records (every step is in the molecule's notes):

- **Counter-ions.** The largest fragment is kept and the others are named in the notes and in the
  `removed_fragments` column (`[Na+]`, `2 x [Br-]`, a second copy of the parent, a co-crystallized acid).
  Bonds to alkali and alkaline-earth metals are ionic and are disconnected first (`CC(=O)O[Na]` is
  sodium acetate). Turn off *keep largest fragment* to compute the whole record instead.
- **Metal salts.** A bondless Al³⁺, Zn²⁺, Fe²⁺/Fe³⁺ or Cu²⁺ cation (as well as Na⁺, K⁺, Ca²⁺, Mg²⁺,
  Ba²⁺, Sr²⁺) is a counter-ion when the rest of the record is an organic ion and counter-ions: fosetyl-Al
  (`[Al+3]` + 3 ethyl phosphonates) computes one ethyl phosphonate anion, and `removed_fragments` lists
  `[Al+3]` and the two other anions. "An organic ion" means its charge is not on carbon (carboxylate,
  phosphonate, sulfonate, phenolate, ammonium); the other fragments may be metal cations, copies of the
  kept ion, inorganic ions (halide, sulfate, ...), water of hydration or other such organic ions. Al, Fe,
  Cu and Zn bonds drawn in the record are never broken (`standardize.SALT_METALS`).
- **Metal complexes.** A removed fragment that holds any other metal, or a salt metal next to fragments
  that are not counter-ions, fails the molecule with `UnsupportedStructure: metal complex: keeping the
  largest fragment (...) would remove Co (...)`: a neutral ligand (CO, a phosphine, a corrin), a carbanion
  (cyclopentadienide, cyanide) or a neutral kept fragment. A carbonyl complex, a cobalamin or a ferrocene
  drawn as separate ions would otherwise be computed as CO, a bare corrin or cyclopentadiene. The batch
  continues.
- **Organometallics drawn with their bonds** (tributyltin chloride, phenylmercury acetate,
  bis(tributyltin) oxide) are computed whole: bonds to metals other than group 1/2 are kept as drawn
  (RDKit's default metal disconnector would turn R-Hg-OAc into R-Hg+ and TBTO into Bu3Sn+). Conformers
  come from ETKDG + UFF; `auto` refines them with GFN2-xTB.
- **Charge.** With protonation on, the parent is neutralized and protonated for the pH. When the charge
  differs from the record's (`input_charge` column), a note says so: `charge +0 as drawn, -1 computed
  (counter-ions removed, then neutralized and protonated for pH 7)`. Two basic sites within three bonds
  (piperazine, a 1,2-diamine, a guanidine's two NH₂) get one proton, not two; a 1,3-diamine gets two.
  With `--no-protonate` the charge of the kept fragment as drawn goes to the engine unchanged (a −2
  phosphonate is computed at −2, an ammonium at +1; `tests/test_phases.py` checks it from SD records to
  the calculator).
- **Structure keys.** Every row of summary.csv and descriptors.csv has `inchikey` and `canonical_smiles`
  of the computed species (after counter-ion removal and protonation) and `input_inchikey` of the record
  as drawn (all fragments), standard InChIKeys from RDKit. Charge states differ in the last letter
  (`-N` neutral, `-M` one proton removed, `-O` one added), so join on the first 14 characters to match
  an anion to its acid. When InChI cannot represent a species the key is empty and a note says why.
- **Ids from SD files.** `--id-col CAS_RN` uses that SD property; a record without it takes its title
  line as the id (then DTXSID/ID, then `<file>_<record>`), and its notes say so.
- **Spin.** The multiplicity (`multiplicity` column) is 1 for closed shells; radicals drawn in the
  record give the high-spin multiplicity (2 unpaired electrons: triplet) and an odd electron count
  without drawn radicals a doublet, both with a note (radicals in databases are often a missing H or
  charge). An explicit multiplicity that cannot fit the electron count fails the molecule.
- **Not a structure.** R groups, `*` and molfile atom lists (`M  ALS`) fail with "not a definite
  structure".
- **xTB polarizability** of a multiply charged anion in the gas phase can diverge (unbound HOMO); a value
  above 10 Å³ per heavy atom (or with negative eigenvalues) is discarded with a note.

dxtb's GFN2 and its field properties (xTB polarizability, IR intensities) need the libcint backend.
On Windows and macOS that is the dxtb fork with tad-libcint 0.4 (FORK-1; Linux can use either):

```bash
pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1" "tad-libcint>=0.4"
```

`engines.libcint_status()` (shown by `quantabench-cli doctor` and the GUI as the reason `gfn2-xtb` is
unavailable) tells a missing tad-libcint from an incompatible one, e.g. tad-libcint 0.4 next to PyPI's
dxtb 0.4 and tad-mctc 0.7. See [INSTALL.md](INSTALL.md#dxtbs-libcint-backend-gfn2-polarizability-ir-intensities).

## Memory guard for dxtb, and the per-molecule time limit

dxtb differentiates through its SCF with its default `scf_mode="full"`: autograd keeps the matrices of every
SCF iteration, so a step's memory grows as (SCF iterations) x nao², where nao is the number of basis functions.
The polarizability (autograd of the dipole with respect to the field) costs about 7x the energy + forces.
In the 325-molecule stress sample this killed four lane processes at 2-3.9 GB. Measured on 2026-11-05 (dxtb 0.4.0,
CPU float64, peak above the ~0.58 GB of a process with torch and dxtb loaded):

| molecule | atoms | nao | charge | GFN1 E+F | GFN1 polarizability | GFN2 E+F | tblite GFN1 polarizability |
|---|---|---|---|---|---|---|---|
| PFOA anion (C8) | 26 | 102 | -1 | 45 MB | 287 MB | | |
| C20 PFCA | 62 | 246 | 0 | 108 MB | 644 MB | | |
| C20 PFCA anion | 62 | 246 | -1 | 237 MB | 1508 MB | | 25 MB |
| Ala10 | 103 | 308 | 0 | 212 MB | 1480 MB | 567 MB | 35 MB |
| Ala10, NH3+ N-terminus | 105 | 312 | +1 | 217 MB | 1551 MB | | |
| Ala15 | 153 | 458 | 0 | 372 MB | | 1522 MB | |
| Ala25 | 253 | 758 | 0 | 920 MB | | | 153 MB |

The full table is in `chem/memory_guard.py`. Charges and the charge dipole without a gradient take 27-122 MB.
Batched refinement (`--batched on`, GPUs) holds B conformers: B x one. Before each dxtb step the pipeline estimates

    MB = k[step] x scf_factor x nao² x conformers + 30

The coefficients are k = 0.0025 for GFN1 energy + forces, 0.0075 for GFN2, 0.016 for the GFN1 polarizability
and 0.0003 for the charges and for tblite's polarizability. They are the upper envelope of the neutral
measurements. `scf_factor` is 1 for a neutral closed-shell molecule, 1.5 for a cation, 2.5 for an anion
(1.7-2.3x was measured), and 6 for an open-shell molecule or one with a transition metal (the SCF can run to
dxtb's 100 iterations). The estimate is compared with the **budget** of the process, minus what the process
already uses. The budget is `resources.memory_budget_gb` (`build --memory-budget GB`). The default 0 means the
memory available when the run starts divided by the number of lanes, and never less than 1.5 GB. A value
below 0 turns the guard off. The run log states the budget.

| step | estimate within the budget | over the budget |
|---|---|---|
| GFN1 charges + dipole (properties step) | dxtb | the whole xTB properties step is skipped, with a note; the molecule stays ok |
| GFN1 polarizability (properties step) | dxtb autograd | **tblite** GFN1-xTB finite field if tblite is installed: the same quantity (Ala10 39.1012 vs 39.1009 Å³). A note says so and `xtb_polarizability_method` = `tblite GFN1-xTB finite field`. Without tblite (or if tblite's estimate does not fit either) the polarizability is skipped, with a note. Charges and dipole stay dxtb's |
| dxtb refinement (GFN1/GFN2), sequential | runs | `MemoryGuardError: ... refinement not started by the memory guard: estimated X GB ...`. Method `auto` tries its next engine and notes why. An explicit method fails the molecule with that reason, and the batch goes on |
| dxtb refinement, batched | B conformers at once | batches small enough to fit (the results do not depend on the batch size), and the error above if not even one fits |

Every note is in `summary.csv`. dxtb's `scf_mode="implicit"` is not used as a cheaper path. It needs 4-5x less
memory for energy + forces, but its forces differed by 0.83 eV/Å from the full mode on the C20 PFCA anion, and
its polarizability came out 0. The frequencies stage (a dxtb Hessian) is not guarded yet.

**Time limit per molecule.** `resources.molecule_timeout_s` (`build --molecule-timeout S`) defaults to 1800 s;
0 turns it off. The slowest molecule of the stress sample that finished took 429 s (a Cr complex in `auto`), so
30 min only ends molecules that are stuck, and a stuck molecule costs at most 30 lane-minutes of a
14,000-molecule batch. QM stages and CREST can take longer per molecule: raise the limit for those runs. A
molecule past the limit is recorded failed with `time limit (1800 s) in <stage>`. This is not a crash, so it is
not tried again in the run; `--retry-failed` (with a higher limit, or 0) redoes it.
- **Lanes:** the parent knows when each lane got its molecule. Past the limit it kills the lane with the
  programs the lane started, records the molecule, and restarts the lane if work is left.
- **One process** (`--lanes 1`, the single-molecule GUI run): best effort. The molecule stops at the pipeline's
  next `should_stop` checkpoint: every optimizer step and conformer, between stages, and the polls of external
  programs. A step without a checkpoint (RDKit embedding, a single dxtb SCF, the xTB properties step) runs to
  its end first. Use lanes when a hard limit matters.
- The external-program limit `engines.timeout_s` (one xtb/CREST/MOPAC/Psi4 run) is separate and still applies.

## AIMNet2 and dxtb on GPUs

- **AIMNet2.** aimnet 0.2 computes neighbor lists and D3 dispersion with NVIDIA Warp kernels, which
  run on CUDA and the CPU only. On ROCm (AMD) and MPS (Apple) QuantaBench uses `chem/aimnet_torch.py`
  instead: the same model with the dense neighbor path and D3(BJ) from `tad-dftd3` (the dxtb fork's vendored
  copy, else PyPI's) with aimnet's
  parameters and reference tables. It matches the stock calculator to < 1e-5 eV (CPU and an RX 9060 XT).
  `QUANTABENCH_AIMNET_BACKEND=stock|torch` overrides the choice.
- **"Warp CUDA warning: Could not find or load the NVIDIA CUDA driver".** Warp's C library prints this
  on every process start on machines without an NVIDIA driver, and no Warp setting turns it off.
  QuantaBench initializes Warp itself before aimnet does (`resources.init_warp_quietly`) and drops only
  that line; other Warp messages still show. Verified on Linux; on Windows it depends on Warp's DLL
  sharing Python's C runtime (if the line still shows there, it is harmless).
- **aimnet neighbor-list bug (fixed in QuantaBench, 2026-11-05).** aimnet 0.2's adaptive neighbor list
  shrinks after a small molecule and only grows again on an overflow error that nvalchemiops 0.5 never
  raises. In one process, a larger molecule evaluated after a small one silently lost neighbors (with the
  real weights on NOVA's CPU: PFOA 3.2 eV and PFOS 7.6 eV off). `aimnet_torch.guard_neighbor_lists` makes
  every stock calculator QuantaBench builds check and grow its lists. **Multi-molecule AIMNet2 runs on
  the CPU made before this fix should be re-run.** (The Warp-free GPU path does not use these lists.)
- **dxtb.** `chem/torch_compat.py` runs the dense linear algebra a GPU build lacks (Cholesky and solve
  on AMD's Windows ROCm wheel; everything, in float64, on MPS) on the CPU inside a dxtb call. Results
  match the CPU, but dxtb 0.4 is 5-8x slower on NOVA's GPU than on its CPU (host synchronizations in its
  overlap integrals), so `dxtb_device()` keeps dxtb on the CPU unless Settings > Compute > "dxtb on the
  GPU" (or `QUANTABENCH_DXTB_GPU`) is `on`. `QUANTABENCH_LINALG=auto|cpu|device` chooses what is routed.

## Finding programs

QuantaBench looks, in this order, at: Settings > Engines (a path, or `wsl:/full/path` on Windows),
the `QUANTABENCH_XTB` / `QUANTABENCH_CREST` / `QUANTABENCH_MOPAC` / `QUANTABENCH_PSI4_PYTHON`
environment variables, PATH, the running Python environment, and conda/mamba environments in the usual
places (`~/miniforge3/envs/*`, `~/mambaforge`, `~/miniconda3`, `~/micromamba`, ...).

Ready-made environments are in [`envs/`](../envs): `tools.yml` (xtb, CREST, MOPAC; Linux/WSL),
`tools-windows.yml` (xtb, MOPAC; Windows) and `psi4.yml` (Psi4 with the pins below). Keep Psi4 in its
own environment.

Each run gets a scratch folder (Settings > Engines > Scratch folder; default `%TEMP%\QuantaBench` or
`/tmp/QuantaBench`). It is deleted on success and kept on failure, and the error message gives its
path. Threads come from the Compute settings; `OMP_STACKSIZE` (default 4G) avoids stack overflows in
xtb and CREST.

## Solvents

`--solvent water` (GUI: Refinement > Solvent) uses one name for every engine: ALPB/GBSA in xtb and
tblite, COSMO (ε) in MOPAC, PCM in Psi4. The 16 names are in `chem/solvents.py` (water, methanol,
ethanol, octanol, acetonitrile, acetone, DMSO, DMF, THF, dichloromethane, chloroform, toluene, benzene,
hexane, diethyl ether, ethyl acetate). Engines without a solvent model run in the gas phase with a note.

Which steps of a run use the solvent (the default model of each engine; summary.csv: `solvent`,
`solvation_model`, and `phase_notes` with one entry per step, e.g. `refine: ALPB(water); ...;
frequencies: ALPB(water)`, or the reason a step ran in the gas phase):

| Step | tblite-gfn2/gfn1 | xtb-gfn2/gfn1, gfnff | MOPAC | dxtb (gfn1/gfn2-xtb), AIMNet2, MMFF |
|---|---|---|---|---|
| Conformer search | RDKit MMFF94s: gas phase in every run (the same pool per phase); CREST: ALPB | same | same | same |
| Refinement | ALPB (GBSA, CPCM on request) | ALPB (GBSA) | COSMO | gas phase |
| Engine properties (dipole, charges, HOMO/LUMO, polarizability, `*_gsolv_kcal`) | ALPB, polarizability included | ALPB (polarizability is D4's α(0)) | COSMO | none |
| xTB properties (`xtb_*`: GFN1-xTB charges, dipole, polarizability) | tblite GFN1-xTB in ALPB | tblite GFN1-xTB in ALPB | tblite GFN1-xTB in ALPB | dxtb, gas phase |
| Frequencies, thermochemistry | ALPB (finite differences of the solvated forces) | ALPB | COSMO (approximate) | gas phase |
| QM stage (Psi4) | PCM where PCMSolver has the solvent | same | same | same |

Without tblite installed, a solvent run takes the xTB properties from dxtb in the gas phase and says so in
the notes and `phase_notes`. In the gas phase the `xtb_*` columns still come from dxtb; dxtb and tblite agree
for GFN1-xTB to 5e-5 e (charges), 1e-4 D and 5e-4 Å³, so gas/water differences are the solvent's.

## Phases: the same input in the gas phase and in water

```bash
quantabench-cli build --file cohorts.sdf --id-col CAS_RN --method tblite-gfn2 --no-protonate \
    --max-confs 50 --window 6 --refine-top 5 --freq --phases gas,water --run-dir RUN
quantabench-cli build --phases gas,water --resume RUN      # continue (or add a phase from a sibling's inputs)
quantabench-cli pair RUN/gas RUN/water --out paired.csv    # or `pair RUN`; works on unfinished runs too
```

`--phases` takes any list of solvent names (`gas` = none). Each phase is an ordinary run folder
(`RUN/gas`, `RUN/water`) with its own job table, run one after the other with `--solvent` set to the
phase, so `--resume`, `--retry-failed`, `--lanes`, `--chunk` and the memory guard work per phase, and
`grid build --solvent water` can compute one phase on other machines (then `merge` the chunks and
`pair` them: several folders of one phase are combined). The two phases may also run at the same time
as two plain builds into `RUN/gas` and `RUN/water` with the same settings. At the end `RUN/paired.csv`
is written.

**Pairing** joins molecules by id **and** `input_inchikey` (the input SMILES for older runs), never by row
order. A row whose id matches but whose input structure differs is not paired (`pair_status`: structure
mismatch). Columns: the identity columns once (`id`, `name`, `input_inchikey`, `inchikey`,
`canonical_smiles`, `input_smiles`, `smiles`, `input_charge`, `charge`, `multiplicity`,
`removed_fragments` and the 2D descriptors), `pair_status` (`ok`, or e.g. `water not computed`,
`gas failed`), `pair_notes`, then `dG_solv_kcal`, `dG_solv_G_kcal`, `dG_solv_vertical_kcal`, then for
every other column of summary.csv and descriptors.csv `gas_<col>`, `water_<col>` and, for numeric ones
where a difference means something, `delta_<col>` = water − gas (no delta for counts, timings and flags).
With three or more phases (or `--reference` other than gas) each phase `p` gets `delta_<p>_<col>` and
`dG_solv_<p>_kcal`, relative to the reference phase.

**ΔG_solv.** tblite and the xtb program report, in a named solvent, the total energy of the solute in the
ALPB continuum: the SCF energy of the polarized solute plus the solvation free energy terms (generalized-Born
polar term, a SASA-based cavity/dispersion/solvent-structure term and an empirical shift), in the `gsolv`
reference state (tblite's and xtb's default). That energy already **is** a free energy of solvation on top
of the gas-phase Hamiltonian, so:

- `dG_solv_kcal` = E_water(R_water) − E_gas(R_gas): each phase's best refined conformer at its own geometry
  (`best_energy_eV`), so it includes the conformer and geometry relaxation between the phases: it is the
  vertical value below plus the gas-phase distortion energy E_gas(R_water) − E_gas(R_gas) ≥ 0. Empty, with the
  reason in `pair_notes`, when the methods, charges, multiplicities or computed species differ, when a
  phase failed, or when the water run fell back to the gas phase.
- `dG_solv_G_kcal` = G_water − G_gas when both phases ran frequencies (`vib_gibbs_kcal`: E + ZPE + thermal
  enthalpy − T·S at each phase's own tightly re-optimized minimum; the water Hessian includes the continuum).
- `dG_solv_vertical_kcal` = the water run's `tblite_gfn2_gsolv_kcal` (or `gfn2_gsolv_kcal` for xtb):
  E_water − E_gas at the water geometry, both SCFs self-consistent; the number the xtb program prints as
  Gsolv plus the polarization cost.

All three are for the transfer from 1 mol/L ideal gas to 1 mol/L ideal solution (Ben-Naim's ΔG\*, the
`gsolv` state). The RRHO translational terms are computed at 1 atm in both phases and cancel. For the
1 atm gas → 1 mol/L solution convention add RT ln(24.46) = +1.89 kcal/mol at 298.15 K (tblite's
`bar1mol` state differs from `gsolv` by exactly this, checked numerically). GFN2-xTB/ALPB `dG_solv_kcal`
(vertical): methanol −4.6 (−4.6; experiment −5.1), acetate −67.4 (−68.0), methylammonium −74.4 (−74.5),
methylphosphonate(2−) −238.8 (−240.3) kcal/mol.

The **solvated polarizability** is the finite-field response with the field and ALPB in one SCF; tblite
supports both together, and the dipole-derivative value equals −d²E/dF² of the same calculations to
1e-4 Å³ (`tests/test_properties_solvent.py`). In water, at the same MMFF geometry, the GFN2 value rises
15-40 % (PFOA and PFOS anions 15 %, 6:2 FTOH 18 %, ethanol 27 %, piperazine 32 %, benzene 42 %) and the
GFN1 `xtb_*` value 8-16 %.

Note that the MMFF94s pre-screen keeps conformers within `--window` + 4 kcal/mol before the `--refine-top`
lowest are refined, and the refined ensemble is pruned to `--window`.

## xtb program (`xtb-gfn2`, `xtb-gfn1`, `gfnff`)

- **Install.** Linux/WSL: `conda install -c conda-forge "xtb>=6.7.1"` (or `envs/tools.yml`). Windows:
  `envs/tools-windows.yml`, or the zip from https://github.com/grimme-lab/xtb/releases; point
  Settings > Engines > xtb at `bin\xtb.exe` if it is not on PATH. xtb ≥ 6.5 is required; 6.7.1 is
  tested. 6.4.x is rejected.
- **Optimizer.** `--optimizer native` uses xtb's own optimizer, 4-10x faster than ASE stepping xtb (one
  extra single point at the end, because `xtbout.json` after `--opt` holds the start geometry's dipole). The
  level is the loosest xtb level whose threshold guarantees the requested fmax (0.03 eV/Å -> `vtight`).
- **Properties.** `gfn2_dipole_debye`, `gfn2_gap_ev`, `gfn2_d4_polarizability_A3` (the D4 dispersion
  model's molecular α(0), a good static-polarizability estimate: water 1.41 vs 1.45 Å³ experiment),
  and with a solvent `gfn2_gsolv_kcal` (E(solvent) − E(gas) at the same geometry) and
  `gfn2_gsolv_term_kcal` (xtb's printed Gsolv term).
- **Pitfalls.**
  - xtb reports an ion's dipole about its coordinate origin. QuantaBench reports it about the centre of
    mass, which matters for carboxylates and sulfonates at pH 7.
  - xtb writes `.xtboptok` and exits 0 even when an optimization ran out of cycles; QuantaBench reads
    the log instead.

## CREST (`--conformers crest`)

- **Install.** Linux/WSL/macOS: `conda install -c conda-forge crest "xtb>=6.7.1"` in one environment
  (`envs/tools.yml`). No Windows build: install it inside WSL2 and set Settings > Engines > CREST to
  `wsl:/home/<you>/miniforge3/envs/qb-tools/bin/crest`. The full path is needed because WSL starts it
  without conda's PATH.
- **Levels.** `--crest-method gfn2 | gfnff | gfn2//gfnff | gfn1`; `--quick` is the default
  (`--crest-full` for a full run). gfn2, gfn1 and gfnff run inside CREST 3. `gfn2//gfnff` needs
  CREST's `--legacy` mode (added automatically) and an `xtb` on the PATH CREST sees.
- **Timings, 1-2 threads.** Ethanol gfnff 2 s; CF3COOH gfn2 38 s; PFBA anion gfnff in water 9 s;
  PFBA gfn2 about 2 min. For C8+ PFAS use gfnff and refine with `xtb-gfn2`.
- Conformers whose bonds changed (CREST can find tautomers) are dropped and logged. If CREST is missing
  or fails, the run falls back to RDKit ETKDGv3 with a note.

## tblite (`tblite-gfn2`, `tblite-gfn1`)

- **Install.** Linux/macOS: `pip install tblite` (part of the `xtb` extra). Windows: PyPI has no wheel,
  so the `xtb` extra skips tblite there. Install the wheels from Brandon's fork (FORK-2) into the venv:

  ```powershell
  pip install tblite --only-binary tblite --find-links https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1
  ```

  They are tblite 0.8.0 for CPython 3.11-3.13 (win_amd64), built in GitHub Actions
  ([workflow](https://github.com/brandonbtm29/tblite/blob/qb/windows-wheels/.github/workflows/qb-windows-wheels.yml))
  with MSYS2's MinGW-w64 gfortran, OpenMP and ddX (CPCM). The only change to upstream is in a build file
  (`config/meson.build` also links the GFortran runtime on Windows). delvewheel bundles the GCC runtime
  and reference BLAS/LAPACK DLLs, the same BLAS as the PyPI Linux wheel. On BTM NOVA they reproduce
  the Linux wheel's water energies (GFN2, GFN1, ALPB, GBSA, CPCM) to 1e-14 Eh, and
  `tests/test_eng_tblite.py` passes. This is not an official tblite release: drop it once upstream
  ships Windows wheels. conda-forge's `tblite-python` works too, but only if QuantaBench runs in that
  conda environment, and its win-64 build has no ddX, so no CPCM.
- Matches the xtb program to 4e-9 Eh. Solvation: ALPB (default), GBSA (falls back to ALPB where GBSA
  lacks parameters), CPCM (ε).
- **Polarizability.** `tblite_gfn2_polarizability_iso_A3` is the electronic response from finite fields
  (6 SCFs), in the run's solvent when it has one (see Phases above). The minimal basis makes it about a
  third of experiment (water 0.37 vs 1.45 Å³): use it for trends.
- **Solvation energy.** With a solvent, `tblite_gfn2_gsolv_kcal` = E(solvent) − E(gas) at the best
  conformer (one more SCF) and `tblite_gfn2_solvation` names the model used (`ALPB(water)`).
- Threads: OpenMP reads `OMP_NUM_THREADS` when tblite is first imported. QuantaBench sets it at startup.

## MOPAC (`mopac-pm7`, `mopac-pm6`)

- **Install.** `conda install -c conda-forge mopac` (Windows: `Library\bin\mopac.exe`) or the
  installer from https://openmopac.net. MOPAC ≥ 22 (open source, no license key); 23.2.5 tested.
- **Energy is the heat of formation in eV**, because MOPAC's gradients are those of ΔHf. Compare
  energies only within one method. ΔHf already includes ZPE and 298 K thermal terms, so after frequencies
  use the corrections (`vib_zpe_kcal`, `vib_h_corr_kcal`, `vib_g_corr_kcal`), not the absolute H and G.
- `--optimizer native` (MOPAC's own optimizer) is 3-15x faster than ASE stepping MOPAC.
- **Properties.** `pm7_heat_of_formation_kcal`, dipole, HOMO/LUMO/gap, ionization energy,
  `pm7_polarizability_iso_A3` (POLAR, closed shell; water 1.26, methanol 2.99 Å³) and, with COSMO,
  `pm7_cosmo_area_A2` and `pm7_cosmo_volume_A3`.
- MOPAC exits with code 0 on errors; QuantaBench reads its error messages instead.
- **COSMO is not smooth:** energies and gradients jump as surface segments switch. Optimizations in COSMO
  can stop short of fmax, and COSMO frequencies are approximate (CF3COO⁻ in water: a −28 cm⁻¹ spurious
  mode). Run MOPAC frequencies in the gas phase.

## Psi4 (QM stage)

```
quantabench-cli build "OC(=O)C(F)(F)C(F)(F)C(F)(F)F PFBA" --method xtb-gfn2 --solvent water \
    --conformers crest --crest-method gfnff --qm wb97x-d/def2-svp --cubes
```

- **Install** in its own environment: `conda env create -f envs/psi4.yml`. Then set Settings > Engines
  > Psi4 Python to that environment's `python` (`QUANTABENCH_PSI4_PYTHON` works too). Without it,
  QuantaBench looks for a `psi4` launcher in the usual places.
- **Levels** are `method/basis`: `wb97x-d/def2-svp` (Spartan's ωB97X-D; dispersion built in),
  `b3lyp-d3bj/def2-tzvp`, `hf/6-31g*`. Use `def2-svpd` (diffuse functions) for anions where it matters.
- **What the stage does** for the `qm_top` lowest conformers: a single point or an optimization (optking).
  For the best one, optionally:
  - properties: dipole about the centre of mass, quadrupole, Mulliken/Löwdin/MBIS charges, HOMO/LUMO;
  - the static polarizability: analytic CPSCF for HF/LDA/GGA/hybrids including ωB97X-D; finite field for
    meta-GGAs, VV10, double hybrids and PCM;
  - density and ESP cubes with surface descriptors;
  - frequencies with IR intensities and qRRHO thermochemistry, computed with QuantaBench's own
    vibrational code.
- **Descriptors.** `qm_energy_ev`, `qm_dipole_debye`, `qm_homo_ev`, `qm_lumo_ev`, `qm_gap_ev`,
  `qm_polarizability_iso_A3`, `qm_mbis_*`, `qm_zpe_kcal`, ..., `qm_surf_*`.
- **Files.** `<id>_qm_log.out`, `<id>_qm_density.cube`, `<id>_qm_esp.cube`, `<id>_qm_geometry.xyz`
  and the QM IR spectrum, written next to the other results.
- **Timings, 2 CPU threads, ωB97X-D/def2-SVP.** Water 1 s. PFBA anion in PCM water: SCF 31 s,
  properties 21 s, cubes 23 s, finite-field polarizability 142 s (PCM forces finite fields).
- **Pitfalls.**
  - PCM solvents: PCMSolver has no DMF, n-hexane, diethyl ether, octanol or ethyl acetate; those QM stages
    run in the gas phase with a note.
  - libint 2.13 makes Psi4 1.9.1 segfault: pin `libint=2.9`.
  - pint ≥ 0.25 breaks qcelemental 0.30, so every optimization fails: pin `pint<0.25`. The driver
    patches this for its own process and warns.
  - `-d3bj`/`-d4` functionals need `dftd3-python` / `dftd4-python`.
  - PCM has no analytic gradients in Psi4: optimizations and frequencies run in the gas phase;
    energies and properties are PCM single points (noted).
  - DFT frequencies by finite differences cost 2(3N−6)+1 gradients (≈145 for PFOA).

## Surfaces (Surfaces panel, `qm_surf_*` descriptors)

From the QM density and ESP cubes: the 0.002 au density isosurface (area, enclosed volume) with the ESP
mapped on it, and the Murray-Politzer GIPF statistics: V_min, V_max, means, σ²₊, σ²₋, σ²_tot, ν, Π.
Without QM cubes, the panel draws a vdW or solvent-accessible surface colored by the point-charge ESP
of the current conformer's charges. Colors follow Spartan and GaussView (red = negative). Water at
ωB97X-D/def2-SVP: 20.5 Å³ and 36.8 Å² at 0.002 au.

The descriptors use the 0.002 au surface (Spartan's), recorded as `qm_surf_iso_au`. Published
Murray-Politzer GIPF values are on 0.001 au, where the ESP is softer (water HF/6-31G*: σ²tot 345 vs 478
(kcal/mol)² at 0.002 au), so compare with the literature only at the same isovalue.

![ESP on the 0.002 au density surface of trifluoroacetic acid](img/esp_surface.png)

## Importing other programs' results

`quantabench-cli import FILE...` and File > Import calculation output read Gaussian, ORCA, Psi4, xtb,
MOPAC and other outputs through cclib (`pip install cclib`, the `interop` extra). Geometries, energies,
charges, frequencies and IR intensities become a built molecule; the Spectra panel shows and animates
its modes. Experimental spectra (CSV, JCAMP-DX including compressed forms) can be overlaid.

### Spartan documents (`.spartan`)

Spartan '24 documents go through the same command and menu item, but all documents given become **one
table**: one row per molecule and document, keyed by the molecule label (the CAS number in the PFAS QSAR
batches), so a project split into many documents needs no manual recombination:

```bash
quantabench-cli import batches/*.spartan --table all.csv --sdf all.sdf     # --conformers all: one row each
quantabench-cli import cohorts/*/*.spartan --table all.csv --wide qsar.csv # + one row per label, all levels
```

- Documents from Spartan for Windows (gzipped `Input.gz`, `PropertyArchive.gz`) and for macOS (the same
  streams uncompressed) both read. A molecule that cannot be read is counted under a named reason (no Input
  stream, Input not readable, no geometry in Input) and listed; nothing is skipped silently.
- Every row has `source_file`, `molecule_index` (position in its document), `storage` (`M0001`), `label` and
  `duplicate_label` (1 when the molecule's own label occurs in more than one molecule; nothing is overwritten),
  then `label_resolved`, `label_source`, `label_note`, `structure_key`, `inchikey`, `canonical_smiles`,
  `structure_charge`, `charge_check`, the level of theory (`spartan_version`, `spartan_task`, `spartan_method`,
  `spartan_basis`, `spartan_solvent`, `spartan_solvent_raw`, `spartan_status`), `formula`, `smiles`,
  `n_atoms`, `charge`, `multiplicity`, `n_conformers`, and every scalar Spartan property as `spartan_<name>`.
- QuantaBench columns, from quantum-chemical calculations only (units checked against Spartan's output, its
  manual and identities between the archive's values): `homo_ev`, `lumo_ev`, `gap_ev` (Spartan: hartree),
  `dipole_debye`, `zpe_kcal`, `h_corr_kcal` (H - E), `g_corr_kcal` (G - E), `entropy_cal_molK`. CPK area and
  volume, ovality, the ESP areas, Spartan's estimated polarizability, hardness, electronegativity, logP and
  the energies (a heat of formation for PM6, a total energy for DFT) stay `spartan_*`; their units are in
  `chem/io_spartan.py`. In this long table a PM6 and a DFT HOMO are both `homo_ev`, told apart by
  `spartan_method` next to them.
- **Labels.** Spartan keeps a molecule's label (the CAS number in the PFAS QSAR batches) only in the document
  it was built in; extracted conformers and the PM6/DFT documents made from them say `M0003` or `M0001[2]`.
  Those rows are joined to their base document (`C1B_P5_SemiEmp_PM6.spartan` -> `C1B_P5.spartan`, which must
  be among the inputs) by structure: elements, connectivity and charge must match exactly one label there.
  `label_source` is `input`, `structure`, `ambiguous` (several base labels share the structure, e.g. an anion
  under the CAS numbers of its salts; `label_note` lists them) or `unresolved` (no base document, or the
  structure is not in it). Nothing is guessed.
- **Structure keys** QuantaBench can match: `inchikey` and `canonical_smiles` of the structure as Spartan drew
  it (its bonds and bond orders, formal charges from each atom's bond-order sum: a one-bond carboxylate O is
  O⁻, a quaternary N is N⁺; R/S and E/Z from the geometry), computed by the same function as QuantaBench's own
  species (`chem.validate.species_keys`). `structure_charge` is the drawing's charge and `charge_check` says
  whether Spartan calculated with it (`ok`) or not (`mismatch`: e.g. a quaternary ammonium calculated as -1,
  a phosphate dianion as -1; their Spartan values are not the drawn ion's).
- `spartan_solvent` is QuantaBench's solvent name (`WASSER` -> `water`); `spartan_solvent_raw` and the level
  text keep Spartan's keyword.
- **`--wide`** writes one row per resolved label with each level of theory's best result (completed first,
  then lowest energy) as prefixed columns: `conf_*` (conformer search), `mm_*`, `pm6_*`, `dft_*`, ... plus
  `<prefix>_level`, `<prefix>_n_rows` (e.g. conformers at that level) and `<prefix>_n_failed`. The prefix
  comes from the method and task, not the document name; when a run has two levels with one prefix, each
  gets its level appended (`conf_dlff03`, `conf_mmff`), so a column never mixes levels. The command prints
  the prefixes it used. An **ambiguous structure** gets one row of its own keyed by its structure:
  `label_resolved` empty, `label_source` = ambiguous, `cas_candidates` = the CAS numbers it could be
  (`;`-separated). Comparing a QuantaBench run with this table: [VALIDATION.md](VALIDATION.md).
- Conformer libraries (conformer searches): `n_conformers` and the lowest-energy saved conformer's values
  (`spartan_conf_energy_kcal`, `spartan_conf_boltzmann_weight`, ...; Spartan's kcal/mol).
- Geometry: the `Input` coordinates. After a Spartan optimization (PM6 `OPT`) these are still the starting
  geometry (the optimized one is not read yet); such rows say so in `spartan_notes`.
- `--sdf` writes the reported geometries (bonds from Spartan) with the label and source columns, ready for a
  QuantaBench run; per-atom Spartan values (e.g. MMFF charges) are SDF atom properties.
- Documents are read molecule by molecule (olefile, pure Python), so memory stays flat (the staged sample of
  7 documents, 2,750 molecules, 102 MB: 10 s on 2 cores, under 200 MB).
- In the GUI, every molecule of a document becomes one result with its saved conformers ranked.

## Not available yet

- NMR shieldings (Psi4 has none; planned through PySCF).
- Orbital isosurfaces (cubes of selected MOs) and local ionization potential maps.
- CREST on native Windows without WSL.
