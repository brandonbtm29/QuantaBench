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
