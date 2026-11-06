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
configuration), and, for Spartan rows without any key, by id when the formulas agree. The CAS id then
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
| q min / q max | e | min/max of the best conformer's charges | `<p>_spartan_q_minus_mul` / `_q_plus_mul` (most negative/positive Mulliken charge) | correlation |

- Gas-phase QuantaBench values pair with Spartan's gas-phase quantum levels (PM6 at its PM6 geometry); ALPB
  water values with the water levels (B3LYP/6-311+G** C-PCM single points at the MMFF geometry, ωB97X-D/6-31G*).
- **Correlation only** where the quantities differ: Spartan's CPK surface is the van der Waals surface, not the
  solvent-accessible one, and its volume uses other radii; `EST_POLARIZ` is an estimate (≈ 45-60 for molecules
  whose computed polarizability is ≈ 5-15 Å³); charge partitions differ between methods. Errors are listed, but
  read r, ρ and the fit.
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
