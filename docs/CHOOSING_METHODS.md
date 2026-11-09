# Choosing methods: the review before a run

Before a run starts, QuantaBench reviews the molecules you are about to build. It looks at what kinds of atoms,
charges and groups are in the batch, checks which methods can handle them, and recommends settings. Every
recommendation says what the method is for, what it corrects, and which measured result it rests on. The point is
the one Spartan never made easy: knowing which model to choose and why.

The numbers below are QuantaBench's own measurements ([VALIDATION.md](VALIDATION.md): the DFT conformer test,
RANK-1 and RANK-1b), not rules of thumb. Each method is explained in the [methods guide](METHODS.md).

## Getting the review

- **In the app.** Press **Run all**, **Run selected** or F5: the Review window opens first. Nothing is blocked:
  **Run anyway** (or **Run**, when there is nothing to warn about) starts the build with the pipeline settings as
  they are, and **Cancel run** drops it. **Review...** next to the Run buttons, or **Build > Review molecules...**
  (F6), opens it at any time. **Build > Review before each run** turns the automatic review off and on.
  - **Batch** (left): what is in the list. Open a count to see its molecules; double-click a molecule to select it
    in the Molecules list.
  - **Findings** (right), most severe first: a red icon for a problem, a yellow one for a warning, a blue one for a
    note. **Apply** sets the finding's recommendation in the pipeline pages and reviews again; **Explain** opens
    the methods guide at the entry to read. The pane below says why, with the measured result and links to every
    molecule and guide entry.
  - **Presets**: Fast screen, Recommended and Careful (below). The tooltip and the line under the buttons give the
    reasons and the estimated time.
- **On the command line.** `quantabench-cli review` takes the same inputs and options as `build`:

  ```
  quantabench-cli review --file cohort.csv --phase water
  quantabench-cli review "OC(=O)C(F)(F)C(F)(F)C(F)(F)F PFBA" --phase gas --method aimnet2
  quantabench-cli review --file cohort.csv --qm b3lyp/def2-svp --json
  ```

  `build` and `grid build` print a short version (the problems and warnings, each with the option that fixes it)
  before they start; `--no-review` skips it.

A review of 14,000 molecules takes a few seconds the first time; after an Apply, a fraction of a second.

## What the review looks at

Everything is counted for the species the run will compute, after the pipeline's standardization: the largest
fragment (when **Keep largest fragment** is on) and the charge state at the run's pH (when protonation is on).

| Item | How it is decided |
|---|---|
| Elements | of the computed species, hydrogens included |
| Charge class | anion (net charge below 0), cation (above 0), zwitterion (net 0 with separate + and − sites; a nitro group's charge pair does not count), neutral |
| Open shell | radicals drawn in the structure, or an odd electron count (multiplicity above 1) |
| PFAS | the OECD 2021 structural definition: at least one fully fluorinated methyl or methylene carbon, i.e. a saturated carbon with two or more F and no H, Cl, Br or I attached (−CF3, −CF2−) |
| Perfluoroalkyl chain | the largest connected group of saturated carbons that carry F and no H, Cl, Br or I (PFOA, C7F15−COOH: 7; PFOS: 8; GenX: 3, the ether O breaks it) |
| Functional groups | SMARTS (`chem/advisor.py`, `GROUPS`): carboxylic acid/carboxylate, sulfonic acid/sulfonate, sulfate ester, sulfonamide, phosphate/phosphonate, alcohol, ether, ester, amide, neutral amine, ammonium, quaternary ammonium, aromatic ring, nitro, thiol/thioether, Cl/Br/I |
| Pure hydrocarbon | C and H only, neutral |
| Size | heavy atoms and rotatable bonds (RDKit's strict count), with their distributions |
| Macrocycle | a ring of 12 or more atoms |
| Undefined stereo | stereocentres and double bonds that could be specified but are not |
| Several fragments | salts, counter-ions, mixtures, and what the run removes |
| Isotopes, metals, other unusual elements | isotope labels; metals; elements outside H, C, N, O, F, P, S, Cl, Br, I |

## Which methods can take which molecules

Each method, basis set and CREST level has an element, charge and spin coverage, taken from its own documentation
and checked where QuantaBench can: the table is in [METHODS.md, "Which molecules each method takes"](METHODS.md).
The review lists how many molecules each one takes and whether it is installed. It never recommends a method a
molecule falls outside of. The ones that matter most:

- **AIMNet2** takes 14 elements (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I) and **closed shells only**. Method
  `auto` picks it by elements and does not check the spin, so the review flags open-shell molecules that would go
  to it.
- **GFN1/GFN2-xTB and GFN-FF** cover H to Rn. **g-xTB** covers H to Lr but is not an engine yet.
- **MMFF94s** covers C, H, N, O, F, Si, P, S, Cl, Br and I (and a few free metal ions); anything else is
  pre-optimized with UFF. Force fields have no electrons, so they do not model a radical's spin.
- **PM7 and PM6-D3H4X** (MOPAC) cover H to La and Lu to Bi.
- **QM basis sets** as Psi4 ships them: def2 sets H to Rn without Ce to Lu; 6-31G* H to Kr; 6-31+G* and 6-311+G**
  H to Ca only; aug-cc-pVDZ H to Ar and Sc to Kr.

## Ranking conformers: a decision table

Picking the lowest conformer is where cheap methods differ most, and every descriptor taken from "the lowest
conformer" (the dipole, shape, Boltzmann averages) follows that choice. Picks are counted within 4 kJ/mol of the
DFT reference's lowest conformer. A random pick is the baseline: a method no better than random gives you an
arbitrary conformer.

| Molecules | Phase | Rank with (today) | Avoid as the ranker | Evidence |
|---|---|---|---|---|
| Pure hydrocarbons | gas or water | GFN2-xTB, or any cheap method | – | RANK-1b: GFN2-xTB 12 of 12 in both phases; every cheap method 10-12 of 12 (their pools span little) |
| Neutral molecules, polar (O, N, S, P, Si, halogens, neutral PFAS) | gas | AIMNet2 (`auto` takes it where its elements allow) | GFN-FF, MMFF94s, PM6-D3H4X | RANK-1b against r2SCAN-3c: AIMNet2 39 of 40 non-fluorinated and 18 of 20 PFAS; GFN2-xTB 34 of 40 and 15 of 20. RANK-1 (Essential 76 against ωB97X-D/6-31G*): AIMNet2 45 of 53, GFN2-xTB 36, a random pick 25 |
| Neutral molecules, polar | water | GFN2-xTB/ALPB for geometries; where conformer-dependent descriptors matter, a QM re-rank in PCM (below) | GFN2-xTB/ALPB's own ranking | RANK-1b: GFN2-xTB/ALPB 16 of 28 polar organics and 14 of 20 neutral PFAS (0.6-0.7 of a random pick's regret); g-xTB + GFN1-xTB CPCM 26 of 28 and 19 of 20 |
| PFAS ions | water | the QM stage on a wide pool: ωB97X-D/def2-SVPD in PCM on the lowest 8, window 15 kcal/mol | GFN2-xTB (any solvent model), PM7, MMFF94s, GFN-FF | RANK-1: GFN2-xTB/ALPB 4 of 17 against ωB97X-D/def2-SVP, a random pick about 5; RANK-1b: 0 of 13 against r2SCAN-3c, regret 36 kJ/mol (1.6 times a random pick's). GFN2-xTB puts short anionic O···F contacts 15-20 kJ/mol each too low. DFT test: DFT's lowest sat at GFN2 rank 2-6, 5-62 kJ/mol above GFN2's lowest |
| Other ions (carboxylates, ammoniums) | water | as PFAS ions; their pools are flat, so misses are a few kJ/mol | trusting any cheap ranker on surfactant-like ions | RANK-1b: GFN2-xTB/ALPB 4 of 8 (0.73 of a random pick's regret); no xTB or AIMNet2 variant ranked the carboxylates well. The electrostatics-only PCM reference itself favours extended alkyl chains |
| Any ion | gas | run it in water instead | – | gas-phase ions are unscreened; a multiply charged anion can be unbound |
| Open-shell molecules | any | GFN2-xTB (tblite or xtb, unrestricted) | AIMNet2 (closed-shell model), force fields (spin ignored) | the AIMNet2 model card |

**Coming: the rank step (BACKLOG RANK-2).** g-xTB in the gas phase plus a GFN1-xTB CPCM solvation term ranked best
of every cheap method: 13 of 17 PFAS ions against ωB97X-D/def2-SVP, 5 of 5 against r2SCAN-3c, 12 of 13 PFAS ions
and 38 of 40 neutrals in water in RANK-1b. It costs about 0.1 % of a DFT single point. When it lands, it replaces
the DFT re-rank as the recommendation for water runs, and every descriptor will follow its ranking. Today only the
QM stage's `qm_*` descriptors follow the DFT-lowest conformer; the `xtb_*` and 3D descriptors stay at the
refinement's lowest.

## DFT levels: what each part corrects

| Part | What it corrects | When it matters | Evidence |
|---|---|---|---|
| Dispersion: -D3(BJ), -D4, ωB97X-D's -D, -V | the London attraction between atoms that touch but are not bonded, which most functionals and HF miss | always: folded conformers, stacked rings and touching chain segments come out too high without it | DFT test: D3(BJ) moved plain B3LYP's energy difference between two PFAS-ion conformers by +10 kJ/mol on average (−3 to +26) and took its correlation with the radius of gyration from −0.50 to 0.00. Many Spartan B3LYP and PM6 results had no dispersion correction: that is the cautionary case |
| Diffuse functions: def2-SVPD, 6-31+G*, aug- | the loosely bound extra electron of an anion, which a compact basis squeezes | anions (carboxylates, sulfonates), polarizabilities, ESP surfaces | RANK-1b: ωB97X-D/def2-SVP favoured compact conformers by +15.0 ± 2.4 kJ/mol per Å of radius of gyration for PFAS ions and +7.8 ± 1.0 for other ions, against r2SCAN-3c in PCM. RANK-1: def2-SVPD brought ωB97X-D within 8 kJ/mol MAE of r2SCAN-3c, from 16 |
| gCP, a larger basis | basis-set superposition error: in a small basis, touching atoms borrow each other's functions and bind too strongly | small bases (def2-SVP, 6-31G*), compact conformers | RANK-1b: for neutrals in the gas phase def2-SVP leaned +3.7 ± 1.9 kJ/mol per Å toward compact conformers against def2-TZVP (1-2 kJ/mol across their pools; no pick changed) |
| -3c composites (r2SCAN-3c, HF-3c) | bundle the three: a matched basis, D4 or D3 dispersion and gCP | a DFT reference near triple-zeta quality at about twice ωB97X-D/def2-SVP's cost | RANK-1b's reference; r2SCAN-3c agreed with ωB97X-D/def2-TZVP on 8 neutrals (ρ 0.98, MAE 1.1 kJ/mol). Not selectable in the QM stage yet: Psi4 runs it as `energy('r2scan3c/')` with its own basis |
| PCM, CPCM, COSMO | the solvent's electrostatic screening | ions and polar molecules in water | as QuantaBench sets it up, Psi4's IEF-PCM has no cavity or dispersion terms; RANK-1b: for surfactant-like ions this favours extended chains |
| ALPB, GBSA | screening plus non-polar surface terms (cavity, dispersion with the solvent) | xTB geometries and properties in water | RANK-1: between conformers of PFAS ions, ALPB's solvation differences missed IEF-PCM's by 20 kJ/mol; GFN1-xTB CPCM by 4 |
| SMD | electrostatics plus fitted cavity, dispersion and solvent-structure terms: solvation free energies | partition and solvation free energies | not in QuantaBench |
| PM6-D3H4X's D3H4X | plain PM6's dispersion (D3), hydrogen-bond (H4) and halogen-contact (X) errors | noncovalent contacts at semi-empirical cost | RANK-1: it still overbinds short anionic O···F contacts in the gas phase |
| GFN2-xTB's multipoles | GFN1's charge-only electrostatics: atomic dipoles and quadrupoles describe hydrogen and halogen bonds | geometries | – |

## Solvent: where each step runs

In a water run, not every step sees the water. The review lists it for the run's settings; in summary:

| Step | In water |
|---|---|
| Conformer search, RDKit | gas phase: MMFF94s has no solvent model, so the pool is the same in every phase |
| Conformer search, CREST | ALPB water |
| Refinement | the engine's own model: ALPB (tblite, xtb), COSMO (MOPAC); gas phase for AIMNet2, dxtb and MMFF94s |
| xTB properties | tblite GFN1-xTB in ALPB when the refinement was solvated |
| Frequencies | the refinement's model for tblite, xtb and MOPAC engines |
| QM stage | IEF-PCM single points; QM optimizations run in the gas phase (Psi4 has no analytic PCM gradients) |

## Input hygiene

- **Salts and counter-ions.** With **Keep largest fragment** on (the default), the largest fragment is computed and
  the counter-ions are removed; the review names them. Off, the fragments are computed together, an ion pair in one
  calculation.
- **Metal complexes drawn as ions** (a ferrocene as Fe²⁺ and two cyclopentadienides) cannot be built: draw the
  metal-ligand bonds.
- **Undefined stereocentres.** RDKit embeds an undefined centre at random for every conformer. One centre mixes two
  enantiomers, which is harmless; two or more mix diastereomers, and an undefined double bond mixes E and Z
  isomers, which differ in energy. Define them, or list each isomer.
- **Radicals** in database records are often drawing errors (a missing H or charge).
- **Protonation.** Acids and bases get their charge state at the run's pH from a small rule table. Perfluoroalkyl
  sulfonamides (pKa about 6.5) become anions at pH 7. With protonation off, a record keeps the charges it was drawn
  with, including a proton in the wrong place: one ion of the DFT conformer test was drawn with the proton on its
  sulfonamide N, 183 kJ/mol above the amine-protonated cation.
- **Isotopes** change masses only: frequencies and thermochemistry, not energies.

## Cost

The review estimates the time of each method on this machine from QuantaBench's measured timings, scaled by
molecule size; trust it to about a factor of 2. Lane-hours are on one 4-thread lane of BTM NOVA, where most timings
were measured.

- DFT: ωB97X-D/def2-SVP in PCM takes about 5 min for 495 basis functions, growing as nbf^2.9 (4-6 min for 30 heavy
  atoms, 12-27 min for 54). B3LYP-D3(BJ) takes 0.72 of that, def2-SVPD about 3x, def2-TZVP about 7x and r2SCAN-3c 2x.
- CREST: this machine's fitted model when `quantabench-cli timings --fit` has saved one, else atoms^4 anchored at
  PFBA (about 2 min with GFN2 `--quick` on 2 threads). GFN-FF searches more than 6x faster than GFN2.
- Refinement and RDKit: the PFAS pipeline timings (PERF-1); AIMNet2 is about 10x faster on a GPU than the CPU
  estimate shown.

## The presets

Each preset is a complete set of pipeline settings for the batch in the phase you chose. A preset only uses
methods that take every molecule: AIMNet2 only without open-shell molecules, and the QM stage only when the def2
basis sets cover every element.

| Preset | Gas phase | Water |
|---|---|---|
| Fast screen | RDKit (30), the lowest 10 refined with AIMNet2 (or GFN2-xTB), no DFT | RDKit (30), the lowest 10 with GFN2-xTB/ALPB, no DFT; the conformer choice of polar molecules and ions is rough |
| Recommended | RDKit (50), the lowest 20 with AIMNet2 (or GFN2-xTB), no DFT: neutral dipoles hardly depend on the conformer | RDKit (50), the lowest 20 with GFN2-xTB/ALPB; for polar molecules and ions, a 15 kcal/mol window and ωB97X-D/def2-SVP(D) in PCM on the lowest 8 |
| Careful | CREST at GFN-FF (or 200 RDKit conformers), the lowest 30 refined, ωB97X-D/def2-TZVP (def2-SVPD for anions) on the lowest 3 | CREST at GFN-FF in ALPB water (or 200 RDKit conformers), the lowest 30, ωB97X-D/def2-SVPD in PCM on the lowest 12 |

## Not there yet

- The rank step (BACKLOG RANK-2): g-xTB + GFN1-xTB CPCM as the ranker, and every descriptor from the ranked lowest
  conformer.
- r2SCAN-3c in the QM stage, and SMD solvation.
- Measured rankings for transition-metal compounds and other elements outside the benchmark sets (the review
  checks their coverage only).
