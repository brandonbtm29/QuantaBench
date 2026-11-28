# Force fields for fluorinated liquids and PFAS (FF-1 design note)

2026-11-28. Item FF-1 ("building a force field for PFAS in MD", Brandon, 2026-11-27). What QuantaBench should use to
parameterize PFAS and other fluorinated liquids for liquid-vapour σ (and later aqueous PFAS), what FF-1 built, how
far it reaches, and what is left. User-facing how-to: docs/ENGINES.md, "Liquid-slab MD and surface tension".

## Recommendation in one paragraph

Stay in the OPLS-AA family and extend it only with published, OPLS-AA-compatible group parameters: Watkins &
Jorgensen's perfluoroalkyl types [2], the other fluorine types already in GROMACS' OPLS-AA distribution (fluoro-,
difluoro- and hexafluorobenzene, benzotrifluoride, monoalkyl fluorides, 2,2,2-trifluoroethanol) [1, 6], Pádua's
fluorocarbon/hydrocarbon junction torsions [8], Black et al.'s perfluoroether groups [12] and CL&P's sulfonyl and
carboxylate heads [9-11], all with OPLS-AA's
own transferable charges and geometric mixing. This is the route our own pilots validated (C6F14: σ 12.46 ± 0.40 mN/m
over four runs against Freire's 12.33, +1 %) and the one the fluorinated-liquid literature validates for density, ΔHvap
and σ (below). It is built as the `oplsaa-pf` force field (`md.oplsaa_pf`). It covers the perfluorinated and the
simpler partly fluorinated molecules; it does not cover most functional PFAS (esters, ethers, amides, sulfonamides,
ketones, acids, CHF2 groups), because no published OPLS-AA group parameters exist for the CF2 next to those groups.
For those the next step is a per-molecule charge route inside OPLS-AA (LigParGen's 1.14·CM1A-LBCC [17, 18] or
1.2·CM5 [19]) with OPLS-AA Lennard-Jones and bonded terms, validated on the σ set's liquid densities before any σ
slab. GAFF2 stays out for fluorinated liquids (pilot: σ 2.6x, density +10 %), and OpenFF Sage is a cross-check, not
the route, until a density check on perfluorocarbons says otherwise.

## What the force field has to get right

- **The liquid density first.** σ rises steeply with liquid density along coexistence; a force field that misses
  the density by 10 % cannot give σ (the GAFF2 pilot: density +10 %, σ 32.18 vs 12.33, `pilot_GAFF2_report.md`). A
  1 ns bulk NPT is the cheap screen before a slab.
- **Dispersion summed properly** (pppm/disp or LJ-PME; MD-LR). The force field must have been fitted with a long-range
  treatment the slab can reproduce; OPLS-AA's were fitted with a cutoff plus a homogeneous tail correction, which is
  what full long-range dispersion gives in the bulk.
- **One mixing rule for all components.** OPLS-AA is geometric; GAFF, CHARMM and OpenFF are Lorentz-Berthelot.
  Combining parameters fitted under different rules gives large errors [27]. Every type in `oplsaa-pf` is
  geometric-mixing OPLS-AA.
- **H-F cross interactions.** Mixtures of hydrocarbons and fluorocarbons need an H-F (or C-F) cross term weaker than
  the geometric mean (about 25 % weaker in ε, Song et al. [28]; Pierce et al. used such a term for semifluorinated
  alkanes [13]; the Lisbon group applies 1.035 to σ_HF and 0.77 to ε_HF in hydrogenated/fluorinated mixtures [16]).
  For a pure liquid of one semifluorinated molecule the same intermolecular H-F pairs occur. `oplsaa-pf` does not
  apply such a term (global geometric mixing); see the caveats.
- **For aqueous PFAS (later):** ion and head-group parameters that match the water model's (CL&P and OPLS-AA ions
  were fitted with TIP3P/SPC-type water), and the counter-ion (Na+, K+) from the same family.

## The evidence

### Our pilots (`Runs/md-pilot-2026-11-20/`, NOVA and the MacBook)

| run | force field | σ (mN/m) | bulk density | against |
|---|---|---|---|---|
| A-fine + split test, four runs, 26 ns | OPLS-AA + W&J (`oplsaa`) | 12.46 ± 0.40 | 1.6716 g/cm³ (A-fine) | Freire 2006: 12.33 (+1.0 %) |
| GAFF2 slab | GAFF2 + AM1-BCC | 32.18 ± 2.11 | 1.8364 (+10 %) | 2.6x Freire |
| OpenFF Sage | written (`c6f14_openff_ewald`), never run | | | |

### Published validations of the candidate force fields on fluorinated liquids

| force field | molecules | density | ΔHvap | σ | ref |
|---|---|---|---|---|---|
| OPLS-AA, Watkins & Jorgensen | CF4 to C6F14, branched, c-C4F8, c-C5F10 (MC, 12 Å + tail) | mean unsigned error 1 % (C6F14 1.681 vs 1.675 at 25 °C) | < 3 % (C6F14 7.86 vs 7.51 kcal/mol) | not fitted | [2] |
| same | C6-C9 perfluoroalkanes, slabs, 16 Å + tail | near experiment | | C6F14 11.25 vs 11.36 (Caço); C7-C9 within -1 to +3 % of older data | [3] |
| OPLS-AA + W&J, modified H-F terms | semifluorinated alkanes F(CF2)n(CH2)mH | agrees with the available data | | agrees (abstract) | [13] |
| OPLS-AA: W&J tail + TFE's -CF2CH2OH + Pádua junction | 1H,1H-perfluoroalcohols, n:1 FTOH (2:1-5:1) | within 2 % (Table 7) | 4-11 % high | MD used for structure, σ from soft-SAFT | [14, 15] |
| OPLS-AA-based PFPE (Black et al.) | perfluorodiglyme, perfluorotriglyme | < 0.02 g/cm³ | < 0.3 kcal/mol | | [12] |
| OPLS-AA (GROMACS) | 146 organic liquids incl. fluorobenzenes | OPLS somewhat better than GAFF | | cutoff-biased in [6]; 61 liquids with a proper long-range correction in [7]; LJ-PME in [20] | [6, 7, 20] |
| GROMOS 54A7 (ATB) | 19 PFAS | about 10 % too high | | | [24] |
| GAFF / GAFF2 | general organics; no perfluorocarbon σ or density benchmark found | GAFF density errors larger than OPLS [6] | | | [6]; our pilot |
| GAFF2, Träg & Zahn refit | perfluoro- and mixed alkanes | not reported | | | torsions only [21] |
| OpenFF Sage 2.0 | drug-like organics; no fluorocarbon σ/density benchmark found | ρ(x) of mixtures improved | | | [22] |
| HFE-specific all-atom FFs | HFE-7000 (Zhang et al.), HFE-7000 to 7500 (Aminian & Vinš) | 0.8 % (VLE) | 2.3 % | | [25, 26] |
| refrigerant-specific FFs | R134a (Peguin et al.), HFOs (Wang & Tse, polarizable) | 0.7 % | 2.2 % | in the measured range; MAE 1.36 mN/m | [29, 30] |

(The literature review of 2026-11-22, `SyncThing/Literature Reviews/2026-11-22 MD surface tension cutoffs and force
fields/`, has the wider evidence; this table adds [12, 14, 16, 25, 26, 29, 30] read for FF-1.)

### Options compared

| route | fits our validated C6F14 model | PFAS functional groups | charges | runs locally | verdict |
|---|---|---|---|---|---|
| **OPLS-AA + published fluorinated groups (`oplsaa-pf`)** | identical | only where a paper parameterized the group | OPLS-AA transferable group charges | yes, in process | **build now (FF-1)** |
| OPLS-AA + LigParGen (BOSS typing, 1.14·CM1A-LBCC) [17, 18] | same LJ family | any organic molecule | per molecule, CM1A from AM1 x 1.14 + bond corrections | no: BOSS is closed, the web server sends the SMILES out | the standard way to fill the gaps; needs a local equivalent |
| OPLS-AA + 1.2·CM5 [19] or AM1-BCC | same LJ family | any | CM5 from a Hirshfeld analysis (Psi4) or AM1-BCC (AmberTools) | yes (Psi4, AmberTools) | candidate for the gaps; not published for PFAS liquids, must be validated |
| OpenFF Sage 2.x [22] | Lorentz-Berthelot | any | AM1-BCC | yes (conda env) | cross-check; run the 1 ns C6F14 density first |
| GAFF2 (+ Träg & Zahn torsions) [21] | Lorentz-Berthelot | any | AM1-BCC / RESP | yes | out for fluorinated liquids (pilot) |
| TraPPE-UA | united atom | perfluoroalkanes, few heads | own | partly | no all-atom H/F detail, no PFAS heads |
| CGenFF / CHARMM | Lorentz-Berthelot | many | own | external | no perfluorocarbon σ/density benchmark found |
| GROMOS 54A7 / ATB [24] | different | many | ATB | web | density about 10 % high for PFAS |
| PFAS- or class-specific FFs (Barbosa 2024 [23], Isele-Holder 2014 [31], HFE and refrigerant FFs) | various | one class each | own | parameters in papers | import class by class where they fit OPLS-AA's rules |

## What FF-1 built: `--ff oplsaa-pf` (`md.oplsaa_pf`)

**Scope, most common first in the project's sets.** Perfluoroalkanes (linear, branched, cyclic; exactly `oplsaa`'s
types and terms); semifluorinated chains (a perfluoroalkyl block on an alkyl chain, as in fluorotelomer
backbones); benzene rings with H, F, CF3, CH3, CH2R and Br; monoalkyl fluorides; alcohols, 2,2,2-trifluoroethanol and
the 1H,1H-perfluoroalcohols; open-chain perfluoroethers and perfluoropolyethers (with -CF2CH2OH ends);
perfluoroalkanesulfonate anions (triflate, PFBS ... PFOS) and trifluoroacetate; Li+, Na+ and K+. The liquid slab takes neutral molecules only; the anions and cations are typed for later aqueous work.

**Sources** (every number is stored as its source writes it, with its line; `scripts/check_oplsaa_pf_params.py`
re-reads the three files and checked all 204 entries, 0 mismatches, 2026-11-28; a test with three planted errors
showed it catches them):

| what | source |
|---|---|
| alkanes, perfluoroalkanes (opls_135-140, 961-965), benzene, fluorobenzene (145/146, 718/719) | GROMACS oplsaa.ff (commit bc88adaf4e), as `md.oplsaa` [1, 2, 5] |
| hexafluorobenzene, difluorobenzene, benzotrifluoride, bromobenzene ring types (opls_720-730) | GROMACS oplsaa.ff; added there for Caleman et al.'s liquid benchmark (foyer cites [6]) |
| toluene / ethylbenzene CH3, CH2 (148, 149); alcohols (154, 155, 157-159) | GROMACS oplsaa.ff [1] |
| monoalkyl fluorides (956-960) | GROMACS oplsaa.ff, which marks them "tentative" |
| 2,2,2-trifluoroethanol (160-165) and its CT_4 bonded terms | GROMACS oplsaa.ff ("Duffy's TFE" in [14]) |
| fluorocarbon/hydrocarbon junction torsions | Pádua 2002 [8], as CL&P's il.ff (commit 898c19d5a4, MIT) lines 455-461 carry them |
| perfluoroether groups: ether O, CF3-O, O-CF2-O, CF2(O)C, CF(O)C2, the Zdol -CF2CH2OH end, their bonded terms | Black et al. 2017 [12], as the authors' foyer file PFE.xml (mosdef-hub/forcefield_perfluoroethers, commit f8fa0aff6c; no license file, numbers cited, not copied as a file) |
| sulfonate (triflate) and perfluoroalkanesulfonyl head; trifluoroacetate | CL&P il.ff [9-11] |
| Li+, Na+, K+ | GROMACS oplsaa.ff opls_406-408 (Åqvist [32]) |

**Rules** (in `md/oplsaa_pf.py`):
- **Typing** by explicit environment rules (RDKit), each pointing to its source; an atom no rule covers refuses the
  molecule with the atom and the reason. Nothing is guessed.
- **Charges are never redistributed.** Each atom carries its type's published charge (OPLS-AA's neutral groups),
  and the molecule must add up to its formal charge exactly. One value is derived, by a published rule: Silva et al.'s
  1H,1H-perfluoroalcohols take TFE's parameters for -CF2CH2OH "adjusting the partial charge of the fluorinated carbon
  to maintain the neutrality of the molecule" [14]: +0.326 e (labelled in the molecule's notes).
- **Bonded terms** are looked up in a fixed order: Black et al.'s PFE.xml for a term touching a perfluoroether atom;
  CL&P for a term touching an anion head; W&J's own C-C-C-C torsion for four perfluoroalkyl carbons (also through
  an O-bonded or Silva CF2, which have no C-C-C-C term of their own); GROMACS' TFE terms (bond type CT_4); Pádua's junction torsions for a
  fluorocarbon/hydrocarbon junction; then GROMACS by bond type, exactly as GROMACS looks it up (exact, then
  X wildcards). A term found nowhere refuses the molecule.
- **Perfluoroether torsions have phase angles** (Black et al.'s CFn-O terms), which LAMMPS' `dihedral_style opls`
  cannot hold: such a molecule is written with `dihedral_style fourier`, its OPLS torsions converted exactly
  (K = V/2, d = 0° or 180°; a test checks the energy at every angle). The OpenMM engine reads fourier torsions.
- **One labelled transfer**: the C-C-C-O torsion of a 1H,1H-perfluoroalcohol (CF2-CF2-CH2-OH) is in none of the
  sources (Silva et al. say they took the missing junction torsions from Pádua, whose set has no C-C-C-O term); the
  TFE CH2 is read as a plain CT and GROMACS' hydrocarbon alcohol term (ffbonded.itp line 1607) is used. Those
  molecules are tagged `transferred`.
- **Tier of each molecule** (`TypedMolecule.tier`, `.recipe`, in README.txt and liquid_setup.json): `published`
  (it follows a published recipe: W&J perfluoroalkanes, OPLS-AA alkanes/aromatics/alcohols/alkyl fluorides, the
  semifluorinated W&J + Pádua recipe, TFE, Silva's alcohols, Black et al.'s perfluoroethers, CL&P's anions),
  `composed` (published groups combined in
  a way no paper validated, e.g. an n:2 fluorotelomer alcohol) or `transferred`.

**What stays the same.** For every molecule `oplsaa` types except hexafluorobenzene, `oplsaa-pf` gives the same
types, charges, Lennard-Jones and bonded terms (a test compares them term by term, C6F14 included, and its ff.params
numbers line by line), so the validated C6F14 model is unchanged: the pilots' own ff.params files
(`c6f14_oplsaa_ewald`, `_finemesh`, `_cutoff16`) are identical, line for line, to what `oplsaa-pf` writes (NOVA,
2026-11-28). `oplsaa` stays the default `--ff`.

### Caveats (each also in the code or the molecule's notes)

1. **Hexafluorobenzene.** `md.oplsaa` types every aromatic C-F with fluorobenzene's opls_718/719 (q ±0.22);
   GROMACS has hexafluorobenzene's own opls_720/721 (q ±0.13) and foyer assigns them. `oplsaa-pf` uses 720/721
   (and 727/728 for ortho-difluorobenzenes). The written-but-never-run pilot `c6f6_oplsaa_ewald` used 718/719. The
   validation below runs both.
2. **W&J's F-C-C-F torsion.** Watkins & Jorgensen's Table 5 has F-C-C-F V1 = -2.500, V3 = 0.250 kcal/mol (the project's
   extraction, `Runs/md-plan-2026-11-19/wj2001_oplsaa_perfluoroalkane_params.csv`; CL&P and Black et al.'s PFE.xml
   carry the same V3). GROMACS' oplsaa.ff (and foyer) apply their 1,2-difluoride term instead, which has V3 = 0. The
   pilots ran with GROMACS' term, and `oplsaa-pf` keeps it so that C6F14 stays the validated model. Whether to switch
   to the paper's V3 (a 0.25 kcal/mol threefold term) is a question for Brandon; it would need a new C6F14 check.
3. **No H-F cross term.** Semifluorinated molecules use global geometric mixing. Pierce et al. reduced the H-F
   interaction for their semifluorinated alkanes [13]; without it, attraction between CH and CF segments is too
   strong [28], so density and σ of those liquids may come out high. Adding pair-specific cross terms needs the
   writers to emit `pair_coeff i j` (and OpenMM exceptions): not done.
4. **Monoalkyl fluorides are "tentative" in GROMACS' own file**, and no liquid validation of opls_956-960 was found.
5. **Long perfluoroalkyl chains**: OPLS-AA overpredicts their melting point and σ grows too fast with chain length
   [3]; above about C10 expect crystallization in slabs.
6. **Perfluoroethers mix two F-C-C-F terms**: a term that touches an O-bonded carbon takes Black et al.'s
   (V3 = 1.046 kJ/mol, W&J's paper value), one inside a perfluoroalkyl chain GROMACS' (V3 = 0, as in the pilots).
   Black et al.'s own molecules (perfluorodiglyme, -triglyme, Krytox, Zdol) have no chain C-C-C-C; through an O-bonded
   carbon it takes W&J's perfluoro torsion here.
7. **Anions**: the PFSA combination (W&J chain + CL&P's perfluoroalkanesulfonyl head) is what Damião et al. describe
   for PFSAs in water (per the 2026-11-22 literature review; their own files not checked); longer perfluoroalkanoates
   (PFOA-) are refused because CL&P has trifluoroacetate only and the α-CF2 has no published type.

## Coverage (2026-11-28, NOVA; `Runs/ff-1-2026-11-27/coverage_summary.md`, per-molecule CSVs there)

| set | molecules | `oplsaa` | `oplsaa-pf` | published / composed / transferred |
|---|---|---|---|---|
| fluorinated σ liquids in the SIGMA-4 pool | 93 | 19 | 28 | 28 / 0 / 0 |
| fluorinated σ rows that boil below 25 °C or are solid | 21 | 3 | 5 | 5 / 0 / 0 |
| PFAS QSAR set, fluorinated (qsar-2026-11-19, anions at pH 7) | 3,857 | 75 | 358 | 272 / 66 / 20 |

- σ set: all 24 perfluorinated alkanes and cycloalkanes but perfluorotripropyl/tributyl/trihexylamine, all simple
  fluoroaromatics but octafluorotoluene and 1-fluoronaphthalene, and the 1-fluoroalkanes. Not covered: 30 esters
  (Jasper's fluorinated esters: ester O next to CF2, and most carry an H(CF2)n terminus), 7 ethers (HFEs and the
  perfluoro/omega-H ethers), CHF2 groups (18 molecules' first refusal), chlorine (6), N and P compounds, Novec 649,
  TFA (neutral acid).
- PFAS set: semifluorinated chains (158 molecules) and perfluoroalkanes are most of the gain; then the n:2 FTOHs
  (composed), the n:1 FTOHs (transferred), 12 perfluoroalkanesulfonates. First refusal reasons, most common first:
  nitrogen (1,153), ether/ester/carbonyl/acid O (900), iodine (234), S outside a sulfonate (231), chlorine (212),
  sp2 carbons (194), CHF2 (138), P (119), Si (87).

## Validation in the cloud container (2026-11-28)

Bulk NPT at 298.15 K and 1 atm, 1,700-2,100 atoms, LJ cutoff 12 Å + analytic tail, PPPM Coulomb, 0.3 ns (the
first 0.1 ns dropped), 1 CPU thread; ΔHvap from a 5 ns gas-phase run of one molecule. Numbers and caveats:
`Runs/ff-1-2026-11-27/validation/validation.md`. (Results: in the next commit of the FF-1 PR, when the runs end.)

## Next steps (FF-1 follow-ups, BACKLOG)

1. σ slabs for the covered liquids that have measured σ, as Brandon's decision (proposal in the FF-1 report).
2. A charge route for the groups with no published charges (LigParGen-style 1.14·CM1A-LBCC, or 1.2·CM5 from Psi4),
   OPLS-AA LJ and bonded terms by bond type, gated by the bulk density of the σ set's esters and ethers.
3. CHF2 / HFC / HFE groups from the class-specific force fields above, if their functional forms fit OPLS-AA.
4. The H-F cross term (pair-specific `pair_coeff i j`) for semifluorinated molecules.
5. Aqueous PFAS: a water model matched to the ions (SPC/E or TIP4P/2005 class) and the anions above; PFCA heads.

## References (checked against Crossref, 2026-11-28)

1. Jorgensen, Maxwell, Tirado-Rives. J. Am. Chem. Soc. 1996, 118, 11225-11236. doi:10.1021/ja9621760
2. Watkins, Jorgensen. Perfluoroalkanes: conformational analysis and liquid-state properties from ab initio and Monte
   Carlo calculations. J. Phys. Chem. A 2001, 105, 4118-4125. doi:10.1021/jp004071w
3. Tsige, Grest. Surface tension and surface orientation of perfluorinated alkanes. J. Phys. Chem. C 2008, 112,
   5029-5035. doi:10.1021/jp710678w
4. Freire et al. Surface tension of liquid fluorocompounds. J. Chem. Eng. Data 2006, 51, 1820-1824.
   doi:10.1021/je060199g
5. Jorgensen, Ulmschneider, Tirado-Rives. J. Phys. Chem. B 2004, 108, 16264-16270. doi:10.1021/jp0484579 (aromatic C-F
   types, as foyer cites them)
6. Caleman et al. Force field benchmark of organic liquids. J. Chem. Theory Comput. 2012, 8, 61-74.
   doi:10.1021/ct200731v
7. Zubillaga et al. Surface tension of organic liquids using the OPLS/AA force field. J. Chem. Theory Comput. 2013, 9,
   1611-1615. doi:10.1021/ct300976t
8. Pádua. Torsion energy profiles and force fields derived from ab initio calculations for simulations of
   hydrocarbon-fluorocarbon diblocks and perfluorinated n-alkanes. J. Phys. Chem. A 2002, 106, 10116-10123.
   doi:10.1021/jp025732n
9. Canongia Lopes, Pádua. Molecular force field for ionic liquids composed of triflate or bistriflylimide anions.
   J. Phys. Chem. B 2004, 108, 16893-16898. doi:10.1021/jp0476545
10. Shimizu et al. Molecular force field for ionic liquids V. J. Phys. Chem. B 2010, 114, 3592-3600.
    doi:10.1021/jp9120468
11. CL&P il.ff (Canongia Lopes, Pádua et al.), github.com/paduagroup/clandp, commit 898c19d5a4 (version 2026/10/09;
    triflate's OTF and trifluoroacetate cite PCCP papers by line)
12. Black et al. Perfluoropolyethers: development of an all-atom force field ... J. Phys. Chem. B 2017, 121,
    6588-6600. doi:10.1021/acs.jpcb.7b00891
13. Pierce, Tsige, Borodin, Perahia, Grest. Interfacial properties of semifluorinated alkane diblock copolymers.
    J. Chem. Phys. 2008, 128, 214903. doi:10.1063/1.2924120
14. Silva et al. Vapor pressure and liquid density of fluorinated alcohols. Fluid Phase Equilib. 2016, 425, 297-304.
    doi:10.1016/j.fluid.2016.06.011
15. Silva et al. Detailed surface characterization of highly fluorinated liquid alcohols. J. Mol. Liq. 2020, 300,
    112294. doi:10.1016/j.molliq.2019.112294
16. Duarte, Machacaz, Pires, Eusébio, Morgado et al. Molecular origin of aneotropy and related surface tension
    anomalies in hydrogenated and fluorinated alcohol mixtures. Langmuir 2026, 42, 10485-10495.
    doi:10.1021/acs.langmuir.6c00218
17. Dodda, Cabeza de Vaca, Tirado-Rives, Jorgensen. LigParGen web server. Nucleic Acids Res. 2017, 45, W331-W336.
    doi:10.1093/nar/gkx312
18. Dodda, Vilseck, Tirado-Rives, Jorgensen. 1.14*CM1A-LBCC. J. Phys. Chem. B 2017, 121, 3864-3870.
    doi:10.1021/acs.jpcb.7b00272
19. Vilseck, Tirado-Rives, Jorgensen. Evaluation of CM5 charges for condensed-phase modeling. J. Chem. Theory Comput.
    2014, 10, 2802-2812. doi:10.1021/ct500016d
20. Fischer et al. Properties of organic liquids when simulated with long-range Lennard-Jones interactions. J. Chem.
    Theory Comput. 2015, 11, 2938-2944. doi:10.1021/acs.jctc.5b00190
21. Träg, Zahn. Improved GAFF2 parameters for fluorinated alkanes and mixed hydro- and fluorocarbons. J. Mol. Model.
    2019, 25, 39. doi:10.1007/s00894-018-3911-5
22. Boothroyd et al. Development and benchmarking of Open Force Field 2.0.0: the Sage small molecule force field.
    J. Chem. Theory Comput. 2023, 19, 3251-3275. doi:10.1021/acs.jctc.3c00039
23. Barbosa, Tavares, Striolo. Molecular interactions of perfluorinated and branched fluorine-free surfactants at
    interfaces. J. Chem. Theory Comput. 2024. doi:10.1021/acs.jctc.4c00459
24. Barbosa, Turner. J. Mol. Liq. 2023, 389, 122826. doi:10.1016/j.molliq.2023.122826
25. Zhang et al. A transferable all-atom force field for thermophysical properties of methyl nonafluorobutyl
    ether, 1-methoxyheptafluoropropane and its binary mixture. J. Chem. Eng. Data 2025, 70, 4105-4115.
    doi:10.1021/acs.jced.5c00485 (abstract only)
26. Aminian, Vinš. Molecular simulations of transport properties of polar hydrofluoroethers. J. Mol. Liq. 2023,
    389, 122847. doi:10.1016/j.molliq.2023.122847 (abstract only)
27. Oliveira, Hünenberger. Influence of the Lennard-Jones combination rules ... J. Chem. Theory Comput. 2023, 19,
    2048-2063. doi:10.1021/acs.jctc.2c01170
28. Song, Rossky, Maroncelli. Modeling alkane+perfluoroalkane interactions using all-atom potentials: failure of the
    usual combining rules. J. Chem. Phys. 2003, 119, 9145-9162. doi:10.1063/1.1610435
29. Peguin, Kamath, Potoff, da Rocha. All-atom force field for the prediction of vapor-liquid equilibria and interfacial
    properties of HFA134a. J. Phys. Chem. B 2009, 113, 178-187. doi:10.1021/jp806213w (abstract only)
30. Wang, Tse. Developing efficient and accurate polarizable molecular simulation models for studying refrigerants and
    their binary mixtures. J. Phys. Chem. B 2026, 130, 8052-8060. doi:10.1021/acs.jpcb.6c03275 (abstract only)
31. Isele-Holder, Ismail. Atomistic potentials for trisiloxane, alkyl ethoxylate, and perfluoroalkane-based surfactants
    with TIP4P/2005 ... J. Phys. Chem. B 2014, 118, 9284-9297. doi:10.1021/jp502975p
32. Åqvist. Ion-water interaction potentials derived from free energy perturbation simulations. J. Phys. Chem. 1990, 94,
    8021-8024. doi:10.1021/j100384a009
