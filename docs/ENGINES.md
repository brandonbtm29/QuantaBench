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
- **Metal complexes.** A removed fragment that holds any other metal fails the molecule with
  `UnsupportedStructure: metal complex: keeping the largest fragment (...) would remove Co (...)`. A
  carbonyl complex, a cobalamin or a ferrocene drawn as separate ions would otherwise be computed as CO,
  a bare corrin or cyclopentadiene. The batch continues.
- **Organometallics drawn with their bonds** (tributyltin chloride, phenylmercury acetate,
  bis(tributyltin) oxide) are computed whole: bonds to metals other than group 1/2 are kept as drawn
  (RDKit's default metal disconnector would turn R-Hg-OAc into R-Hg+ and TBTO into Bu3Sn+). Conformers
  come from ETKDG + UFF; `auto` refines them with GFN2-xTB.
- **Charge.** With protonation on, the parent is neutralized and protonated for the pH. When the charge
  differs from the record's (`input_charge` column), a note says so: `charge +0 as drawn, -1 computed
  (counter-ions removed, then neutralized and protonated for pH 7)`.
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
  (6 SCFs). The minimal basis makes it about a third of experiment (water 0.37 vs 1.45 Å³): use it for
  trends.
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
```

- Every row has `source_file`, `molecule_index` (position in its document), `storage` (`M0001`), `label` and
  `duplicate_label` (1 when the label occurs in more than one row; nothing is overwritten), then `formula`,
  `smiles`, `n_atoms`, `charge`, `multiplicity`, `n_conformers`, and every scalar Spartan property as
  `spartan_<name>`. A Spartan value takes a QuantaBench column only when it is the same quantity in the same
  units; the mapping and the units are documented in `chem/io_spartan.py`.
- Conformer libraries (conformer searches): `n_conformers` and the lowest-energy saved conformer's values
  (`spartan_conf_energy_kcal`, `spartan_conf_boltzmann_weight`, ...; Spartan's kcal/mol).
- `--sdf` writes the reported geometries (bonds from Spartan) with the label and source columns, ready for a
  QuantaBench run; per-atom Spartan values (e.g. MMFF charges) are SDF atom properties.
- Documents are read molecule by molecule (olefile, pure Python), so memory stays flat: a 100-molecule
  conformer-search batch takes ~0.7 s and the process stays under 100 MB.
- In the GUI, every molecule of a document becomes one result with its saved conformers ranked.

## Not available yet

- NMR shieldings (Psi4 has none; planned through PySCF).
- Orbital isosurfaces (cubes of selected MOs) and local ionization potential maps.
- CREST on native Windows without WSL.
