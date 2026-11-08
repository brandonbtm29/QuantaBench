# Validation against Spartan (VAL-1)

QuantaBench's GFN2-xTB (and AIMNet2) descriptors are checked against Brandon's Spartan '24 results: PM6
gas-phase optimizations with QSAR properties and B3LYP/6-311+G** (some ωB97X-D/6-31G*) single points in
C-PCM water. Three steps:

```bash
# 1. Spartan documents -> one wide table (one row per CAS label or ambiguous structure, prefixed per level)
quantabench-cli import cohorts/*/*.spartan --table spartan.csv --wide spartan_qsar_wide.csv
# 2. a reproducible, stratified validation set (CSV of SMILES + id + cohort that build --file reads)
quantabench-cli sample spartan_qsar_wide.csv -o val300.csv            # -n 300 --seed 20261105 (defaults)
# 3. QuantaBench in the gas phase and in ALPB water, then the comparison
quantabench-cli build --file val300.csv --method tblite-gfn2 --no-protonate --run-dir runs/val_gas
quantabench-cli build --file val300.csv --method tblite-gfn2 --solvent water --no-protonate --run-dir runs/val_water
quantabench-cli validate runs/val_gas runs/val_water --spartan spartan_qsar_wide.csv --out runs/val_report
```

`--no-protonate` keeps the species Spartan calculated (the sample's SMILES carry its charges); with `--ph 7`
QuantaBench would run its own charge states, which then join to Spartan's rows of that state if there are any.
Code: `chem/validate.py` (module docstring) and `chem/io_spartan.py`; tests: `tests/test_validate.py`
(`QB_SPARTAN_WIDE=/path/to/wide.csv` adds a check on a real table).

## Structure keys and joins

Both sides are keyed by the same function, `chem.validate.species_keys`: the standard **InChIKey** and RDKit's
**canonical isomeric SMILES**. The Spartan side is the structure as Spartan drew it (its bonds and bond orders;
formal charges from each atom's bond-order sum; R/S and E/Z from the geometry), so an anion is keyed as the
anion even where the older `smiles` column was a connectivity-only perception. A wide table written before
these columns existed still works: its keys come from `smiles`, except for rows whose SMILES lacks the
formula's hydrogens or has a charged carbon (perception artefacts of ions; re-import for those).

A QuantaBench molecule joins, in order, by InChIKey, canonical SMILES, the stereo-free InChIKey (first block
and protonation flag: an acid never matches its anion, a SMILES without R/S matches Spartan's 3D
configuration), by where the charge sits for ions (`charge-location`: the neutralized structure's formula and
connections plus the total charge; a sulfonimide anion drawn N- and the same anion drawn O- have different
InChIKeys), and, for Spartan rows without any key, by id when the formulas agree. The CAS id then
cross-checks each structure join (`id_check`): `same`, `differs` (the id is Spartan's label of another
structure: another protonation state or a wrong entry, listed in the report), or `not in Spartan`.
QuantaBench's own `inchikey`/`canonical_smiles` columns are used when a run has them; otherwise they are
computed from `smiles` (or the run's SDF).

**Charge check.** `charge_check = mismatch` marks Spartan rows calculated with another charge than the
structure drawn (in the staged C3A sample: quaternary ammonium, sulfonium and phosphonium cations calculated
as -1, phosphate dianions as -1; 12 of 47 molecules). They still join, but give no values. The sampler leaves
them out (`--allow-charge-mismatch` keeps them).

**Ambiguous structures** (one structure under several CAS numbers of a base document, e.g. an anion entered as
the acid's and its salts' CAS numbers) have their own wide row: `label_resolved` empty, `label_source` =
ambiguous, `cas_candidates` = the CAS numbers. A QuantaBench molecule of that structure joins it, and its id
cross-checks against the candidates.

## The sampler (`quantabench-cli sample`)

- Eligible: a structure key; completed Spartan values (`--require any` = PM6 or DFT-water, `pm6`, `dft`,
  `both`, `none`); elements within AIMNet2's (H B C N O F Si P S Cl As Se Br I, `--elements`), so every
  molecule can also be refined with AIMNet2; charge classes (`--charges neutral,anion,cation`); 1 to 60 heavy
  atoms (`--min-heavy`, `--max-heavy`); one fragment; Spartan's charge fitting its drawing. One molecule per
  structure (the row with the most Spartan values).
- Strata: cohort (the document name's prefix, `C1A_`, `C3A_`, ...; the quantum level's document first, archived
  documents last; `--cohorts FILE` maps labels to cohorts instead) x charge class (`--stratify`, also `size`).
- Quotas: N split evenly over the strata, a small stratum's remainder going to the others (water filling).
- Within a stratum: molecules with DFT-water and PM6 values first, then DFT only, then PM6 only (`--prefer`),
  each tier shuffled with the seed and spread round-robin over heavy-atom bins (`--size-bins 10,20,35`).
- Output: `id, smiles, name, cohort, charge_class, charge, inchikey, n_heavy, size_bin, has_pm6,
  has_dft_water, label_source, cas_candidates`, and `<name>.design.json` with the filters, quotas and counts.
  The same table, options and seed always give the same sample.

On the staged table (14,837 rows, keys from `smiles`), the default sample is 300 of 11,838 eligible
molecules: 34 each of C1A, C1B, C2A, C4A, C4B, C5A neutral, 33 C2B neutral, 33 C3A anions, all 18 C3A cations
and all 12 C3B anions; 237 with PM6, 64 with DFT-water, 1 with both. DFT-water values exist almost only for the
charged cohorts, which have no PM6, so the solvation-shift comparison needs PM6 or DFT runs on the same
molecules (open question in BACKLOG VAL-1).

## The comparison (`quantabench-cli validate`)

Inputs: run folders (`summary.csv` + `descriptors.csv`, per-molecule `<id>_summary.json` for charges),
merged tables (`merge`), or the paired gas/water table (`gas_*`, `water_*`, `delta_*`; QuantaBench computes
water - gas itself and warns when a `delta_*` column has the other sign). A gas run and a water run of the
same input list become one record per id.

Like with like, in QuantaBench's units:

| property | unit | QuantaBench | Spartan (per level) | compared by |
|---|---|---|---|---|
| dipole | D | `<engine>_dipole_debye` (e.g. `tblite_gfn2_`), `dipole_debye_best` (the refinement method's: AIMNet2, GFN2 charges, ...), `xtb_dipole_debye` | `<p>_dipole_debye` | errors |
| HOMO, LUMO, gap | eV | `<engine>_homo_ev`, `_lumo_ev`, `_gap_ev` | `<p>_homo_ev` (Spartan's hartree x 27.211386), `_lumo_ev`, `_gap_ev` | errors |
| polarizability | Å³ | `<engine>_polarizability_iso_A3`, `xtb_polarizability_iso_A3` | `<p>_spartan_est_polariz` (Spartan's empirical estimate) | correlation |
| surface area | Å² | `sasa_A2_best` (1.4 Å probe) | `<p>_spartan_cpkarea` (CPK, van der Waals) | correlation |
| volume | Å³ | `volume_A3_best` (RDKit vdW grid) | `<p>_spartan_cpkvolume` (CPK) | correlation |
| CPK area | Å² | `cpk_area_A2_best` (Spartan's radii, no probe) | `<p>_spartan_cpkarea` | comparison |
| CPK volume | Å³ | `cpk_volume_A3_best` (Spartan's radii) | `<p>_spartan_cpkvolume` | comparison |
| q min / q max | e | min/max of the best conformer's charges | `<p>_spartan_q_minus_mul` / `_q_plus_mul` (most negative/positive Mulliken charge) | correlation |

- Gas-phase QuantaBench values pair with Spartan's gas-phase quantum levels (PM6 at its PM6 geometry); ALPB
  water values with the water levels (B3LYP/6-311+G** C-PCM single points at the MMFF geometry, ωB97X-D/6-31G*).
- QuantaBench's QM stage (`qm_dipole_debye`, `qm_homo_ev`, ...: source `qm`) pairs the same way. Spartan's Est.
  Density Functional level (`dlxd2xv`, a neural-network energy) counts as a gas-phase quantum level for the
  dipole only: its `dlxd2xv_dipole_debye` is the ωB97X-D/6-31G* SCF's (`dlxd2xv_qm_dipole_level`), so
  `gas_dipole_qm_vs_dlxd2xv` compares like with like at the level, not at the geometry (VAL-2 below).
- **Correlation only** where the quantities differ: Spartan's CPK surface is the van der Waals surface, not the
  solvent-accessible one, and its volume uses other radii; `EST_POLARIZ` is an estimate (≈ 45-60 for molecules
  whose computed polarizability is ≈ 5-15 Å³); charge partitions differ between methods. Errors are listed, but
  read r, ρ and the fit.
- **`sasa_A2` before 0.3.0.dev3 is wrong.** FreeSASA was given zero atom radii, so the value was the area of bare
  probe spheres. Tables of the 2026-11-05 validation run have the old value; rebuild, or compare `cpk_*`, which
  only a new run has.
- **Solvation shifts** (a molecule with gas and water values): QuantaBench water - gas against Spartan's water
  level minus PM6 gas, flagged `shift (different levels)`: Spartan's difference mixes methods and geometries.
- Spartan values of failed jobs are left out (`--include-failed` keeps them).

Statistics per pair, overall and per cohort and charge class: N, MAE, RMSE, mean signed error (QuantaBench -
Spartan), Pearson r, Spearman ρ, the line QuantaBench = slope x Spartan + intercept (R²), and the top-k
outliers by |error| (`--top`). Outputs in `--out`:

- `validation.csv`: one row per matched molecule, `<pair>__spartan`, `<pair>__qb`, `<pair>__diff`
- `join.csv`: every QuantaBench molecule's join, id check and notes (unmatched ones too)
- `summary.csv`: the statistics (`group_type` all / cohort / charge_class)
- `report.md`: join coverage, id/structure disagreements, the statistics table, outliers, units and caveats
- `parity_<pair>.png`: Spartan on x, QuantaBench on y, by charge class (needs matplotlib:
  `pip install "quantabench[validate]"`; skipped without it)

A 10-molecule check (GFN2-xTB with tblite, 3 conformers, gas and ALPB water) against the staged table joined
all 10 by InChIKey with their CAS ids; GFN2's HOMO tracks PM6's (r = 0.96, 1.3 eV lower), its LUMO and gap do
not (MAE 4.7 / 3.5 eV), and the dipoles agree to about 1.2-1.5 D MAE (N = 8, and the optimizations stopped
at the test machine's 10-step limit: a plumbing check, not a result).

# VAL-2: the QM stage against Spartan's ωB97X-D/6-31G* (Essential 76 PFAS)

The question: does QuantaBench's Psi4 stage reproduce Spartan at Spartan's level, and which level should the
4,700-PFAS descriptor run use?

## What Spartan computed

The "Spartan EPA Essential 76 PFAS" folder (128 documents, 1.4 GB, February 2026), imported 2026-11-06:

- **The main document** (76 neutral molecules, the CompTox "Essential" list): one multi-step conformer
  distribution per molecule (`MS_CONFP2`). The steps are:
  1. an MMFF94 search (40 kJ/mol window, at most 200 conformers);
  2. DLFF03 energies;
  3. two optimizations with DLFFG1, Spartan's neural-network estimate of the ωB97X-D/6-31G* geometry;
  4. ωB97X-D/6-31G* gas-phase single points (at most 30 conformers within 10 kJ/mol);
  5. DLXD2XV, the neural-network estimate of ωB97X-V/6-311+G(2df,2p)//ωB97X-D/6-31G*, which ranks the
     conformers and gives their Boltzmann weights;
  6. DLNMR1 NMR shifts.

  63 molecules completed. 13 failed: Spartan's neural networks cover only closed-shell neutral molecules of H,
  C, N, O, F, S, Cl and Br, which leaves out 6 iodides, 3 silanes, 2 phosphonic acids and the sodium salt
  2806-15-7. PFOS (45298-90-6) was drawn as the anion but calculated as a neutral molecule (`charge_check` =
  mismatch).
- **`Conformer NMR, Est DFT/`** (63 documents, 1,006 conformers): the saved conformers of each search, labelled
  `Molecule1`, `Molecule41`, ... Each one has the ωB97X-D/6-31G* energy (`SPENERGY`, hartree), the dipole
  vector (`QM_DIPOLE`, debye), the quadrupole, the DLXD2XV energy (`POSTENERGY`) and NMR shieldings. The lowest
  `POSTENERGY` conformer is the main document's geometry (to 1e-6 Å) and has its values.
- **`Conformer Distribution/`** (64 documents, 4,890 conformers): an earlier DLFF03 search with force-field
  energies only. One extra: perfluorooctane (307-34-6) also has 59 MMFF/water optimizations and one
  ωB97X-D/6-31G* C-PCM water optimization with Spartan's QSAR properties (HOMO/LUMO, ESP extrema and polar
  areas, density-surface area and volume, CPK, EST_POLARIZ).

So Spartan's ωB97X-D/6-31G* reference for the 63 molecules is **gas-phase energies and dipoles per conformer**,
at its neural-network geometries. HOMO/LUMO, ESP extrema, 0.002 au areas and volumes and polarizabilities at
that level exist for one molecule, across all 1,067 of the PFAS QSAR documents. Spartan's DFT QSAR properties
otherwise exist only at B3LYP/6-311+G** C-PCM water (the charged cohorts, VAL-1).

## The import

```bash
quantabench-cli import "Spartan EPA Essential 76 PFAS"/**/*.spartan --table essential76_long.csv \
    --wide essential76_wide.csv --sdf essential76_best.sdf
```

- 5,972 molecules give 76 wide rows. Every `MoleculeN` conformer is resolved to its CAS label by structure;
  before `MoleculeN` was treated as a Spartan default label, there were 847 rows.
- Prefixes:
  - `conf_dlxd2xv`: the 63 searches;
  - `conf_hf`: the 13 failed searches;
  - `dlxd2xv`: the Est. DFT single points, best conformer by `POSTENERGY`, with `dlxd2xv_dipole_debye`,
    `dlxd2xv_qm_dipole_level` = `WB97X-D/6-31G(d)` and `dlxd2xv_spartan_spenergy`;
  - `mm_dlff03`;
  - `mm_mmff_solvent_wasser` and `dft` (perfluorooctane only).
- InChIKeys:
  - 65 equal the CompTox export's;
  - 10 differ only in the stereo layer: Spartan's geometry gives R/S where CompTox gives none, and `validate`'s
    stereo-free join matches them;
  - 2806-15-7 has none, because Spartan drew the Na bonded to two O.
- `charge_check` is `ok` for 74 molecules. PFOS is the mismatch.

## Pairs

| Spartan (wide table / per conformer) | QuantaBench | same geometry | where |
|---|---|---|---|
| `dlxd2xv_dipole_debye` (ωB97X-D/6-31G* SCF, best conformer) | `qm_dipole_debye` (`--qm wb97x-d/6-31g*`, gas) | fixed-geometry runs: yes; `build`: no | script; `validate` (`gas_dipole_qm_vs_dlxd2xv`) |
| `dlxd2xv_spartan_spenergy` (hartree) | the QM energy | fixed-geometry runs | script |
| `SPENERGY` of every conformer | conformer energies (P1E) | yes | script: relative energies, ranking |
| Boltzmann-averaged `QM_DIPOLE` (DLXD2XV weights) | `qm_dipole_debye` | no | script (`spartan_conformers_wb97xd.csv`) |
| perfluorooctane `dft_homo_ev`, `dft_lumo_ev`, `dft_spartan_qsar_elpot_min/max` (kcal/mol), `_polar_area_75/100/125`, `_area`, `_volume` (its density surface) | `qm_homo_ev`, `qm_lumo_ev`, `qm_surf_esp_min/max_kcal`, `qm_surf_esp_polar_area_*_A2`, `qm_surf_area_A2`, `qm_surf_volume_A3` | no (the import keeps the starting geometry of an optimization) | one molecule |

## The pilot (Psi4 1.9.1 through QuantaBench's driver, Spartan's best-conformer geometry, 2 threads)

| molecule | basis functions | ΔE Psi4 - Spartan, exact integrals | ΔE, density fitting (the default) | Δ dipole |
|---|---|---|---|---|
| TFA (76-05-1) | 107 | -1.5e-5 Eh | +5.9e-5 Eh | +0.0001 D |
| triflic acid (1493-13-6) | 126 | | +1.8e-4 Eh | +0.001 D |
| PFPrA (422-64-0) | 152 | +4.6e-5 Eh | +1.3e-4 Eh | -0.0007 D |
| PFHpA (375-85-9) | 332 | | +4.1e-4 Eh | 0.000 D |

The dipole vectors agree in Spartan's frame (angle < 0.04°), and the basis-function counts show that Spartan's
6-31G* uses Cartesian d functions, as Psi4's does. QuantaBench's QM stage therefore reproduces Spartan's
ωB97X-D/6-31G*. The density-fitting error, 0.04-0.26 kcal/mol in total energy, does not affect properties.

The other levels at the same geometry for TFA and PFPrA, against ωB97X-D/6-31G* gas:

| level | dipole | HOMO, LUMO | α |
|---|---|---|---|
| def2-SVP gas | -0.05 to -0.07 D | -0.21 to -0.25 eV | -4 % |
| 6-31G* PCM water | +0.75 to +0.83 D (+26 to +28 %) | HOMO -0.04 eV, LUMO +0.13 eV | +20 % (finite field, reaction field included) |

PCM water also moves the 0.002 au ESP minimum by -7 kcal/mol and the maximum by +14 kcal/mol; the area does not
change.

PFHpA (375-85-9, 23 atoms) at the same geometry:

| level | dipole (D) | HOMO / LUMO (eV) | α (Å³) | 0.002 au area / volume | V_min / V_max (kcal/mol) |
|---|---|---|---|---|---|
| 6-31G* gas | 1.92 | -10.86 / 0.45 | 11.82 | 223.3 / 210.3 | -28.8 / 79.5 |
| def2-SVP gas | 2.07 | -11.12 / 0.20 | 11.37 | 223.1 / 213.2 | -28.6 / 79.0 |
| def2-SVP PCM water | 2.51 | -11.02 / 0.36 | 13.83 | 223.2 / 213.1 | -32.9 / 88.4 |

**Timings.** These are wall times per molecule with 2 threads in the cloud container, a Xeon at 2.1 GHz (the
"full" protocol: SCF, properties, cubes and polarizability):

| protocol | TFA (107 bf) | PFPrA (152) | PFHpA (332) | steps at PFHpA |
|---|---|---|---|---|
| 6-31G* gas | 29 s | 63 s | 533 s | SCF 127, properties 112, cubes 44, analytic α 247 |
| def2-SVP gas | 29 s | 88 s | 507 s | SCF 101, properties 102, cubes 71, analytic α 229 |
| def2-SVP PCM water | 128 s | 113 s | 857 s | SCF 130, properties 56, cubes 50, finite-field α 617 |

- The time grows as nbf^2.5. The 63 molecules have a median of 365 basis functions (6-31G*) and a maximum of
  647.
- On NOVA (3 lanes x 4 threads, taking a lane as 2-4x a cloud job), the P1, P2a, P2b, P2c and P2d set takes
  about 9-19 h (P2b's diffuse basis about 4.5 h of that), or half that with the MacBook taking `--part 2/2`.
  P1E takes about 4-9 h, and `qb` and `qbwater` about 2.5 h each.
- The 4,700-PFAS list has 3,708 SMILES-defined molecules of at most 60 heavy atoms, with a median of 26 heavy
  atoms and about 400 basis functions. On NOVA alone, the full protocol takes about 5-10 days in the gas phase
  (def2-SVP) and 9-17 days in PCM water.
- In PCM, three quarters of the water cost is the finite-field polarizability: six more PCM SCFs. Computing α
  in the gas phase (analytic) or not at all brings the water protocol close to the gas one.

**Psi4 pitfalls found here** (ENGINES.md):
- ESP cubes need density fitting.
- `scf_type pk` filled 12 GB of disk at 332 basis functions.
- One free-atom SCF of the MBIS step failed (triflic acid, def2-SVP), and the Psi4 process then hung after
  writing its result. The driver now cleans up and `run_driver` stops such a process.

## The runs (`scratch/val2/` on NOVA, not in the repository)

- `val2_import.py`: the import above, plus `spartan_conformers_wb97xd.csv`, `essential76_summary.csv` and the
  CompTox inputs `val2_inputs_63.csv` and `val2_inputs_all.csv`.
- `val2_fixed_geometry.py`: the driver at Spartan's geometries.
  - Protocols:
    - P1: ωB97X-D/6-31G* gas, full properties;
    - P1E: every conformer's energy;
    - P1K: P1 with exact integrals;
    - P2a: def2-SVP gas;
    - P2b: def2-SVPD gas;
    - P2c: 6-31G* water;
    - P2d: def2-SVP water;
    - P2e: def2-TZVP gas.
  - It is resumable and shares a list between machines (`--part K/N`).
  - Psi4 runs in local scratch; only small files go to SyncThing.
  - The report compares each level with Spartan and with P1, and fits t = a·nbf^b per protocol.
- `run_val2_nova.ps1` and `run_val2_mac.sh`: the stages `fixed`, `fixedE`, `qb` and `qbwater` (QuantaBench's own
  geometries: RDKit -> AIMNet2 -> `--qm wb97x-d/6-31g* --qm-top 3 --cubes`, and the candidate default GFN2-xTB/ALPB
  -> `--qm wb97x-d/def2-svp` in PCM water), then `validate`.

# VAL-3: whose conformers? (`quantabench-cli conformers compare`)

The question: at Spartan's own geometries GFN2/ALPB reproduces Spartan's B3LYP/C-PCM water dipoles of anions to
0.41 D, yet QuantaBench's water dipoles differ from Spartan's by 4.35 D MAE (61 ions, the 2026-11-06 validation).
Is it the conformer each program picks, and if so, did QuantaBench's search miss Spartan's conformer, or find it and
rank it differently?

```bash
# Spartan's conformers: documents directly, or an import (one row and SDF record per conformer)
quantabench-cli import C3A_P1.spartan C3A_P1_All_Conformers_Top30.spartan C3A_P1_DFT_Top5.spartan ... \
    --conformers all --table c3_long.csv --sdf c3_conformers.sdf
quantabench-cli conformers compare --a c3_long.csv c3_conformers.sdf \
    --include "*_DFT_Top5.spartan" "*_All_Conformers_Top30.spartan" \
    --b Runs/validation-2026-11-06/water/<NOVA share> Runs/<MacBook share> \
    --a-name Spartan --b-name QuantaBench --rescore gas,water --relax water --workers 2 --out cmp_c3_water
# two QuantaBench runs (RDKit vs CREST) the same way
quantabench-cli conformers compare --a Runs/<rdkit water run> --b Runs/<crest water run> --a-name RDKit \
    --b-name CREST --rescore water --relax water --out cmp_rdkit_crest
```

Code: `chem/conformer_compare.py` (module docstring), `cli_conformers.py`; tests: `tests/test_conformer_compare.py`.

## Conformer sets and levels

- **Inputs** (either side): QuantaBench run folders (any machine's share; a folder of run folders one level down),
  `.spartan` documents, or a Spartan import's long table + SDF. Sets are joined by constitution
  (`io_spartan.constitution_key`: elements, connectivity, total charge), so bond orders, where an ion's charge is
  drawn, stereo and atom order do not matter. Sets of one constitution on one side are merged, and the same
  geometry twice (symmetric RMSD < 0.01 Å) becomes one conformer with both levels' energies.
- **`--include`**: the Spartan documents whose conformers count (the others only resolve labels).
- **Levels**, each conformer with all it has (kJ/mol; dipoles in debye):
  - `Spartan B3LYP/6-311+G** water`: SPENERGY of a single point; dipole about the centre of mass;
  - `Spartan DLXD2XV` and `Spartan WB97X-D/6-31G(d)`: Est. DFT; dipole = the SCF's QM_DIPOLE;
  - `Spartan MMFF94 (search)`: DC_LIST_ENERGY_KCAL, which extracted conformers carry;
  - `QuantaBench <method> <phase>`: each kept conformer's energy and dipole;
  - `common GFN2-xTB gas SP`, `... ALPB(water) SP`, `... ALPB(water) opt`: `--rescore` and `--relax`.
- Energies of an optimization (OPT) belong to the optimized geometry, which the import does not read: they are left
  out. Failed jobs give no energies.
- Each side ranks at its own best level (Spartan's DLXD2XV > DFT > semi-empirical/xTB > force field; `--a-level`
  picks another).

## Matching conformers with PFAS symmetry

- **The problem.** A symmetric RMSD minimizes over the molecule's automorphisms. A perfluoroalkyl chain has many:
  CF3 (6) x each CF2 (2) x SO3- (6) gives 4,608 heavy-atom permutations for PFOS and 98,304 for a C12 sulfonate.
  RDKit's `GetBestRMS` aligns every one, and a `maxMatches` cap can miss the best.
- **The factorization.** The permutations come only from **sibling groups**: terminal heavy atoms of one element
  and H count on one parent (the F of CF3/CF2, the O of SO3-/COO-/NO2, the CH3 of N(CH3)3+). These are the
  resonance-equivalent atoms. An acid's =O and -OH differ by their H, so they are not siblings.
  - The **reduced graph** has the other heavy atoms plus one pseudo-atom per group at its centroid. Its
    automorphisms are few: chain reversal of a symmetric molecule, or the two CF3 of a C(CF3)2 branch.
  - Up to 2,000 heavy-atom automorphisms are **enumerated exactly** (one batched SVD). This covers TFA through
    PFOA (768) and equals RDKit's `GetBestRMS` in the tests.
  - Above that, each reduced automorphism is aligned on the reduced points. Each group's atoms are assigned
    optimally, all heavy atoms re-fitted, and the steps repeated from three starts.
  - For alike conformers this is the exact minimum. The tests check it against the full enumeration with noisy,
    rotated copies whose group atoms are renumbered. For conformers > 1 Å apart it can come out a few hundredths
    of an Å high.
  - Cost per pair: PFOS 1.6 ms (exact: 30 ms); a C10 sulfonate (18,432 automorphisms) 2 ms (exact: 120 ms).
  - A skeleton on one line (TFA) always takes the exact path.
- **Torsions** (under the RMSD's atom map):
  - `backbone`: non-ring bonds between skeleton atoms;
  - `rotor`: one end is a symmetric top (CF3, COO-, SO3-, N(CH3)3+); reported only;
  - `polar_h`: an OH/SH hydrogen, which tells a syn acid from an anti one (heavy atoms do not).
- **The same conformer** means:
  - RMSD <= 0.5 Å (`--rmsd`; QuantaBench's own pruning threshold);
  - every backbone and polar-H torsion within 60° (`--torsion`). Gauche and anti differ by 120°, the P/M twist of
    a perfluoro chain by about 30°;
  - the same configuration at every stereocentre (signed volume of its first three heavy neighbors).
- **Mirror images** count as the same conformer: same energy, same |dipole| (`--no-mirror` turns this off).
  Spartan lists P and M helices separately with one MMFF energy. A reflected chiral conformer matches only when
  the two sides drew opposite enantiomers.

## Reports

`--out` gets these files:

- `molecules.csv`: per molecule:
  - the conformers on each side; the levels each side ranks by;
  - how many of A's conformers within 5 and 10 kJ/mol of A's minimum B found, and the reverse;
  - whether the two lowest are the same, with their RMSD, mirror flag, largest and mean backbone torsion
    difference, and the rotor and polar-H differences;
  - where A's lowest sits in B's list, and the reverse;
  - the cause (below), the common-level energies and the dipoles.
- `levels.csv` and `levels_summary.csv`: coverage per side and level, the common levels included.
- `conformers.csv`: every conformer with its relative energies, dipoles and closest partner.
- `summary.csv`: the statistics overall and per charge class, cohort and cause.
- `report.md`: the same as text.
- `rescore_cache.jsonl`: the common-level results by geometry. A rerun reuses it.

**Search or ranking** (`cause`). The common level decides in the phase of B's run. It uses the optimized energies
when both lowest conformers have them, else the single points.

| cause | meaning |
|---|---|
| `same best` | the two lowest conformers are the same |
| `ranking: found, ranked higher` | B has A's lowest but its own energies put another lower: the energies |
| `ranking: missed, higher at common level` | B never had A's lowest, but the common level puts it above B's lowest: B's ranking would not have picked it anyway (the level, not the search) |
| `search: missed, lower at common level` | B never had A's lowest and the common level puts it below B's lowest: B's search missed a conformer its own level prefers |

`union_best` says which side has the common level's lowest conformer of both sets together.

The dipole gap is split two ways:

- the common level at A's lowest geometry against A's own dipole: the method alone, same conformer;
- B's lowest against the common level at A's lowest: the conformer alone.

## Results (2026-11-06, `Runs/conformers-compare-2026-11-06`)

Computed in the cloud on 2 cores. The C3 common level took about 2 h (2,066 conformers: single points and GFN2/ALPB
optimizations); the Essential 76 one about 50 min (1,237).

**C3A/C3B ions in water (61).**
- Spartan's side: the DFT_Top5 and All_Conformers_Top30 documents, taken from an import that NOVA's
  `scratch/cc/extract_c3.py` makes. That gives 1,765 conformers (median 33 per ion). The 5 DFT conformers are not
  all among the Top30: they come from the Top5 search.
- QuantaBench's side: the Nov 6 water run, 235 conformers (at most 5 per ion).

| measure | value |
|---|---|
| same lowest conformer | 6 / 61 |
| Spartan's B3LYP-lowest in QuantaBench's set / QuantaBench's lowest in Spartan's | 10 / 61, 16 / 61 |
| Spartan's conformers within 5 kJ/mol found: B3LYP / MMFF / GFN2-opt ranking | 15/108, 33/228, 24/218 |
| QuantaBench's conformers within 5 kJ/mol found by Spartan | 35/120 |
| RMSD / largest torsion difference of the two lowest (median) | 1.75 Å / 135° |
| cause: same / ranking (found 4, missed but higher at GFN2 34) / search / no GFN2 energies | 6 / 38 / 16 / 1 |
| GFN2's rank of Spartan's B3LYP-lowest among Spartan's conformers (median) | 11th of ~33 (first in 10 of 60) |
| ions where only Spartan's pool has GFN2's lowest; QuantaBench's lowest above it (mean / median) | 49 of 61; 14.8 / 7.8 kJ/mol |
| dipole MAE: QuantaBench's lowest vs B3LYP (Spartan's lowest) | 4.35 D (anions 5.03, cations 2.45) |
| GFN2/ALPB at Spartan's lowest geometry vs B3LYP: method only | 0.52 D (anions 0.47, cations 0.67) |
| share of the gap from the conformer | 88 % |
| GFN2's lowest of both sets together vs B3LYP | 6.2 D (mean signed −4.8 D) |

- The ion whose GFN2 optimizations change bonds is the cation 67940-02-7: a proton moves from N6 to N1.
- 75260-69-4 forms an O-S contact of 2.0 Å in 15 of its 34 optimizations. Those optimizations are set aside.
- The water gap is the conformer, and the conformer is mostly the energy model. GFN2/ALPB puts compact, low-dipole
  ions lowest; B3LYP/C-PCM (single points at MMFF geometries) puts extended ones lowest.
  The DFT conformer test below finds the dispersion-free B3LYP is not the cause: ωB97X-D and B3LYP-D3(BJ) side with
  Spartan's ranking, and GFN2/ALPB is the outlier.
- QuantaBench's search is short at its own level as well (the 5 lowest by gas-phase MMFF94s are refined). Fixing
  only that moves its dipoles further from Spartan's.

**Essential 76 in the gas phase (63).**
- Spartan's side: the Est. DFT documents, 962 conformers. 44 of the 1,006 saved are the same geometry with the atoms
  renumbered (symmetric molecules) and merge.
- QuantaBench's side: a new gas run (`--method tblite-gfn2 --max-confs 50 --window 6 --refine-top 5 --no-protonate`),
  275 conformers.

| measure | value |
|---|---|
| same lowest conformer | 8 / 63 |
| Spartan's DLXD2XV-lowest in QuantaBench's set / QuantaBench's lowest in Spartan's | 10 / 63, 27 / 63 |
| Spartan's conformers within 5 kJ/mol (DLXD2XV) found / QuantaBench's found by Spartan | 36/607, 60/150 |
| cause: same / ranking (2 + 18) / search | 8 / 20 / 35 |
| GFN2's rank of Spartan's lowest (median) | 2nd (first in 23 of 63) |
| QuantaBench's lowest above GFN2's lowest of both sets (mean) | 6.2 kJ/mol |
| dipole MAE vs ωB97X-D/6-31G*: QuantaBench's lowest / GFN2 at Spartan's geometry / GFN2's lowest of both | 1.28 / 0.81 / 1.17 D |

- In the gas phase the search is the larger part, and the dipole gap is half method: GFN2's dipoles are 0.76 D
  above ωB97X-D's on average.
- **Spartan's TFA and PFPrA** documents hold one conformer each, an **anti** acid (H-O-C=O 180° and 177°). GFN2
  puts them 19 and 17 kJ/mol above QuantaBench's syn conformers. The larger carboxylic acids are syn in Spartan's
  documents too.

**CREST** (the MacBook's first `crest-2026-11-06` runs: 10 ions, 10 neutrals, gas and water).
- `crest` exited with code 1 for 18 of 20 molecules in every run; those fell back to RDKit.
- CREST ran for TFA and 3330-15-2 in the gas phase. 3330-15-2 has the same lowest conformer as Spartan; TFA is syn
  where Spartan is anti.
- Comparing a CREST run needs only its folder as `--b`. Copy the Spartan side's `rescore_cache.jsonl` into the new
  `--out`, so Spartan's conformers are not recomputed.

## DFT conformer test (2026-11-07, `Runs/dft-test-2026-11-07`)

The question: Spartan's B3LYP has no dispersion correction. Does dispersion-corrected DFT side with GFN2/ALPB's
lowest conformer or with Spartan's? If ωB97X-D and B3LYP-D3(BJ) side with GFN2 and plain B3LYP with Spartan, the
missing dispersion explains the disagreement and GFN2's ranking is defensible.

**Setup** (BTM NOVA; scripts in `scratch/dft_test/`, not in the repository):
- **Ions:** the 10 with the largest water-dipole errors (`Runs/crest-2026-11-06/molecules.csv`, 30-54 heavy atoms).
- **Pool per ion,** by each source's own ranking:
  - S1-S3: Spartan's B3LYP/6-311+G** C-PCM lowest (DFT_Top5 single points at MMFF geometries);
  - Q1-Q3: QuantaBench's GFN2/ALPB lowest (the validation run; 1-3 conformers for some ions);
  - C1-C3: CREST's (GFN-FF search, GFN2/ALPB refinement; the Mac's `crest-gfnff/water`);
  - G1: the GFN2/ALPB-optimized lowest of Spartan's whole pool (DFT_Top5 + All_Conformers_Top30).
- **One geometry per conformer:** GFN2-xTB/ALPB(water) optimization with `conformer_compare.rescore`
  (fmax 0.03 eV/Å; 362 of 391 from the compare's cache).
  - Every optimization kept its conformer: heavy-atom RMSD ≤ 0.49 Å, backbone torsions ≤ 46°.
  - Two CREST pairs optimized to one conformer and are one entry ("C1+C3", "C2+C3").
- **Levels:** Psi4 1.9.1 single points with the QM stage's settings (density fitting, IEFPCM water with a GePol/UFF
  cavity, dipoles about the centre of mass; SCF converged to 1e-6, the two timing jobs to 1e-8):
  - ωB97X-D/def2-SVP;
  - B3LYP-D3(BJ)/def2-SVP;
  - plain B3LYP/def2-SVP: the same SCF without the D3(BJ) term. Psi4 adds -D3 after the SCF; on trifluoroacetate
    in PCM this equals a separate B3LYP SCF to 1e-13 Eh, and Psi4's D3(BJ) energy equals dftd3-python's to 2e-12 Eh.
  - Diagnostics: GFN2 single points at the same geometries in the gas phase and in CPCM water (tblite, ddX), and
    ωB97X-D in the gas phase for S1, GFN2's pick and the ωB97X-D/PCM lowest of S1, Q1, C1 and G1 (22 single
    points).
- **Cost:** 168 PCM single points (84 conformers × 2) plus 22 gas-phase ones, about 30 lane-hours, none failed.
  They ran on 3-5 lanes × 4 threads at BelowNormal priority: 7.1 h of wall time after a 0.6 h timing run. The
  largest ion (815 basis functions) took 17-27 min per ωB97X-D single point on a lane and 12-19 min per B3LYP one
  (DiskDF at 3 GB per job); the smallest (487-500) took 4-6 min. Time grows as nbf^2.9.
- **Left out:**
  - 67940-02-7: a tautomer case, below.
  - Two CREST conformers of 29811-19-6: C2 has a new O-C bond. In C3 a sulfonate O sits 1.93 Å from the ester
    carbonyl C, a half-formed bond that the 1.25 × covalent-radii rule misses; ωB97X-D puts it 87 kJ/mol up.

**Result** (9 ions; dE = E(S1) − E(GFN2's pick), kJ/mol; negative: the level prefers Spartan's pick):

| ion | q | GFN2's pick | lowest at ωB97X-D | B3LYP-D3(BJ) | B3LYP | dE: GFN2 | ωB97X-D | B3LYP-D3(BJ) | B3LYP | D3 part |
|---|---|---|---|---|---|---|---|---|---|---|
| 75032-81-4 | −1 | Q1 | C1 | C1 | C1 | +29.7 | −52.0 | −42.3 | −68.6 | +26.3 |
| 92340-26-6 | −1 | G1 | S1 | S1 | S1 | +61.9 | −62.5 | −61.1 | −77.5 | +16.4 |
| 57589-85-2 | −1 | G1 | S2 | S2 | S1 | +53.4 | −6.7 | +0.9 | −16.1 | +17.0 |
| 54950-05-9 | −1 | G1 | C1 | C1 | C1 | +30.5 | −42.1 | −42.7 | −45.9 | +3.2 |
| 68815-72-5 | −1 | G1 | C3 | C3 | Q2 | +51.5 | −12.9 | −10.7 | −30.6 | +19.9 |
| 3793-74-6 | −1 | G1 | C1+C3 | C1+C3 | C1+C3 | +31.7 | −39.9 | −40.0 | −51.4 | +11.3 |
| 313475-52-4 | +1 | C1 | S1 | S1 | G1 | +4.9 | −3.1 | −2.0 | +0.9 | −2.9 |
| 29811-19-6 | −1 | G1 | Q1 | S3 | Q1 | +9.4 | +4.4 | +3.4 | +3.7 | −0.3 |
| 61119-62-8 | −1 | G1 | S1 | S1 | S1 | +33.1 | −15.4 | −16.2 | −17.5 | +1.3 |

"D3 part" is B3LYP-D3(BJ) minus B3LYP. Spartan's dipole is its B3LYP/C-PCM dipole of S1 at the MMFF geometry;
"at S1" is each level's dipole at S1's GFN2 geometry.

| level | lowest is GFN2's pick / S1 / other | S1 vs GFN2's pick: preferred / tie (≤ 4 kJ/mol) / not | mean dE | dipole MAE vs Spartan, own lowest | at S1 | own lowest vs ωB97X-D's |
|---|---|---|---|---|---|---|
| GFN2/ALPB | 9 / 0 / 0 | 0 / 0 / 9 | +34.0 | 10.7 D | 2.0 D | 9.5 D |
| ωB97X-D | 0 / 3 / 6 | 7 / 1 / 1 | −25.6 | 7.5 D | 1.3 D | — |
| B3LYP-D3(BJ) | 0 / 3 / 6 | 6 / 3 / 0 | −23.4 | 6.5 D | 1.3 D | 1.0 D |
| B3LYP | 0 / 3 / 6 | 7 / 2 / 0 | −33.7 | 6.3 D | 1.3 D | 2.4 D |

- **Dispersion is not the explanation.** D3(BJ) moves dE(S1 − GFN2's pick) by +10 kJ/mol on average toward GFN2's
  pick (median +11, range −3 to +26). That is 15 % of the 68 kJ/mol between GFN2/ALPB and plain B3LYP. ωB97X-D and
  B3LYP-D3(BJ) side with Spartan's pick about as often as plain B3LYP does.
  - Plain B3LYP does lean toward extended conformers: the median rank correlation of its energies with the radius of
    gyration is −0.50, and D3(BJ) takes it to 0.00.
  - GFN2 leans the other way (+0.50), far beyond what dispersion gives.
- **GFN2's ranking is the outlier.**
  - No DFT level puts GFN2's pick lowest in any ion. ωB97X-D's lowest is Spartan's S1 in 3 ions, S2 in 1, CREST's
    C1 or C3 in 4 and QuantaBench's Q1 in 1.
  - Within a pool, ωB97X-D and B3LYP-D3(BJ) agree: median Spearman ρ 0.95, relative energies 2.6 kJ/mol apart on
    average. GFN2 against ωB97X-D: ρ −0.36, 23 kJ/mol.
  - Our B3LYP/def2-SVP//GFN2 orders Spartan's S1-S3 as Spartan's B3LYP/6-311+G**//MMFF does (median ρ 1.00; GFN2
    −0.20), so the DFT side reproduces Spartan where the two overlap.
- **Hamiltonian and solvent model.** GFN2/ALPB and ωB97X-D/PCM differ by 60 kJ/mol in dE(S1 − GFN2's pick).
  - The gas-phase single points split it into +39 kJ/mol from the Hamiltonian (median +43) and +20 kJ/mol from ALPB
    against PCM solvation (median +25).
  - GFN2 single points in CPCM (ddX) still put GFN2's pick lowest in 5 of 9 ions and never prefer S1.
- **Short O···F contacts.**
  - GFN2's pick packs an anionic or carbonyl O within 2.6 Å of chain fluorines (Bondi sum 2.99 Å) in 7 of 9 ions,
    with three or more such contacts in 5. Spartan's S1 has one in 1 of 9.
  - Relative to the pool, GFN2 puts conformers with three or more such contacts 43 kJ/mol lower than ωB97X-D does;
    conformers without any sit 22 kJ/mol higher.
  - So GFN2 likely overbinds O···F-C contacts in these anions. That is a hypothesis, not tested here: the contacts
    come with compactness, and a DFT optimization would open them somewhat.
- **GFN2 cannot pre-screen for DFT here.** ωB97X-D's lowest sits at GFN2 rank 2-6 of 7-10 (median 5),
  5-62 kJ/mol above GFN2's lowest (median 33). Re-ranking GFN2's lowest 3 would have missed it in 7 of 9 ions.
- **Boltzmann averages do not rescue GFN2.** Its pool-averaged dipoles are 10.7 D from Spartan's, as its lowest
  conformer's are: the gaps are tens of kJ/mol.
- **The dipole follows the ranking level, not the functional.** GFN2's pick is 9.5 D from ωB97X-D's pick on average,
  B3LYP-D3(BJ)'s pick 1.0 D. At the same conformer (S1) the levels differ from Spartan by 1.3-2.0 D.
- **67940-02-7 is a protonation problem.**
  - With the proton on the dimethylamino N (QuantaBench's conformers) the cation is 183 kJ/mol (ωB97X-D) or
    170 kJ/mol (B3LYP-D3(BJ)) below the drawn sulfonamide-NH2⁺ form that Spartan computed. GFN2 gives 197 kJ/mol.
  - The validation's 11.5 D dipole gap for this ion comes from the protonation site, not the conformer.

**Recommendation for the PFAS QSAR run (ions in water):**
- Rank conformers with DFT in PCM water, not with GFN2/ALPB. Neither GFN2's lowest nor GFN2 Boltzmann averages are
  defensible for these ions.
- Re-rank a diverse pool, not GFN2's lowest few: folded and extended conformers from RDKit and CREST, about 6-10 per
  ion. Use `--qm wb97x-d/def2-svp --qm-top N --solvent water` on a pool built for it.
- B3LYP-D3(BJ)/def2-SVP ranks like ωB97X-D (2.6 kJ/mol) and costs about 30 % less.
- One single point takes 4-6 min (30 heavy atoms) to 12-27 min (54) on a 4-thread NOVA lane; time grows as nbf^2.9.
- The QM stage's `qm_*` descriptors come from its lowest conformer, but the xTB descriptors stay at GFN2's lowest.
  The QSAR run needs the descriptors of the DFT-lowest conformer.

**Limits:**
- Ten ions chosen for their large dipole errors: worst cases.
- def2-SVP has no diffuse functions. Anions with an exposed charge may shift by a few kJ/mol.
- IEFPCM is electrostatics only, and ALPB adds non-polar terms; the +20 kJ/mol solvent part depends on the model.
- Single points at GFN2 geometries: no DFT relaxation and no thermal corrections. GFN2's own picks, with their short
  O···F contacts, would relax the most.
- At most 10 conformers per ion. DFT's lowest may lie outside the pool.
