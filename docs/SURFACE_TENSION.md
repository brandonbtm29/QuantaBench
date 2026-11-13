# Surface tension: data sources and a modeling plan

The goal: predict the surface tension of PFAS and other organics with QSAR/QSPR models built on the
electronic and physical descriptors QuantaBench computes (polarizability, van der Waals and solvent-accessible
volume and area, dipole, ESP surface descriptors, charges, HOMO/LUMO, gas and water phases). This page covers
two things: where experimental and predicted values can come from, and how the descriptors should be tied to
them. The PubChem part is implemented (`quantabench-cli pubchem enrich`, docs/ENGINES.md, "PubChem"). The model
is backlog item QSAR-3.

This page describes sources; it does not reproduce their tables. Values that are pulled programmatically stay
in the output folder (never in this repository), with their source and reference on every record.

## Two different targets

"Surface tension of a PFAS" means two different measurements, and the literature mixes them:

| | Pure-liquid σ | Surfactant γ in water (γ_CMC) |
|---|---|---|
| What | the neat liquid against its vapor or air | an aqueous solution against air, usually at or above the critical micelle concentration |
| Depends on | the molecule and T (about −0.1 mN/m per K) | molecule, concentration, counter-ion (H, Na, K, NH4), added salt, pH, purity, T |
| Typical values | alkanes 18-28, alcohols 22-25, water 72, perfluoroalkanes 11-16 mN/m | 15-40 mN/m at the CMC for fluorinated surfactants (water alone: 72) |
| Theory | Sugden parachor, σ = (P·Δρ/M)⁴; corresponding states | Szyszkowski/Gibbs adsorption: γ = γ0 − RTΓmax ln(1 + Kc) |
| Exists for | compounds that are liquid at the temperature | water-soluble surface-active compounds |

The two cannot be pooled in one model. Most PFAS of interest are solids at 25 °C: the perfluoroalkyl acids
with six carbons or fewer are liquids, the longer ones are solids (Kirk-Othmer, quoted by HSDB in PubChem's
PFOA record), and PFSAs, their salts and most precursors are solids. For them only an aqueous γ exists, or a
pure-liquid σ of the melt at a stated temperature above the melting point. PubChem's record for PFOA shows the
problem: HSDB gives "15.2 dynes/cm" without a temperature or concentration, citing Moody & Field (2000), a review
of perfluorinated surfactants in fire-fighting foams. PFOA melts at 54 °C. The value is almost certainly γ of an
aqueous solution at or above the CMC, not the pure liquid's. `pubchem enrich` classifies it as `unknown` and
marks it `needs_review`. The classifier does not guess "aqueous" from indirect evidence.

**Recommendation.** Two targets, two models: (1) pure-liquid σ at 25 °C (or σ(T)) for compounds that are
liquid there, organics and the liquid PFAS (short PFCAs, perfluoroalkanes, fluorotelomer olefins and
alcohols, fluorinated ethers and solvents, refrigerants); (2) γ_CMC (with log CMC) for PFAS surfactants in water,
always with the counter-ion and electrolyte recorded. Model (1) is where QuantaBench's gas-phase descriptors
fit naturally. Model (2) needs the water-phase descriptors and only has a few dozen compounds of data.

## Sources

Access was checked on 2026-11-05. The cloud environment used for this work blocked direct HTTP to
pubchem.ncbi.nlm.nih.gov, api-ccte.epa.gov, comptox.epa.gov, webbook.nist.gov, trc.nist.gov, www.nist.gov,
data.nist.gov and pfas-1.itrcweb.org (connections refused by the proxy). Descriptions of those sources come
from their documentation pages, read through a search tool, and from the captured PubChem responses in
`tests/data/pubchem`. Compound counts marked "about" are from the sources' own descriptions or the literature,
not counted here.

| Source | Pure / aqueous | Temperature | Access | Terms | PFAS coverage |
|---|---|---|---|---|---|
| PubChem (HSDB, ICSC, ...) | both, unlabeled | in the text when stated | PUG REST/View API, free | NLM: public data, cite sources | low: a few PFAS (PFOA: an aqueous-looking value) |
| EPA CompTox Dashboard / CTX API | predicted (TEST): pure liquid; experimental: few | TEST: 25 °C | web, CTX API (free key), batch by DTXSID | US government, public | TEST predictions for nearly all PFAS on the Dashboard; few measurements |
| NIST Chemistry WebBook, fluid properties | pure (saturated liquid) | σ(T) along saturation | web forms, free | NIST SRD terms (no bulk redistribution) | refrigerants and a few fluorinated fluids |
| NIST ThermoML archive | pure and mixtures, labeled | every point has T | free bulk download (data.nist.gov) | publisher-permitted; cite the article | a few: fluorinated ionic liquids, perfluoroalkanes |
| NIST TRC SOURCE / Web Thermo Tables | pure, critically evaluated σ(T) | full T range | subscription (WTT, TDE) | proprietary | some |
| DIPPR 801 (AIChE) | pure, recommended σ(T) (Eq. 106) | Tt to Tc | subscription | proprietary, no redistribution | perfluoroalkanes, some fluorinated compounds; few PFAAs |
| Yaws (Knovel / handbooks) | pure, σ(T) coefficients, many estimated | fitted range | subscription / book | proprietary | many fluorinated compounds, often estimated |
| Jasper 1972, J. Phys. Chem. Ref. Data 1, 841 | pure, evaluated, linear σ = a − bT | stated range | free PDF (NIST SRD JPCRD reprints); coefficients also in the MIT-licensed `chemicals` package (Jasper via Lange's) | AIP/NIST copyright on the compilation | perfluoroalkanes and a few fluorinated liquids |
| Wohlfarth & Wohlfarth, Landolt-Börnstein IV/16 (1997), IV/24 (2008) | pure and binary mixtures | stated | Springer (library) | proprietary | some fluorinated liquids |
| Mulero et al., JPCRD 2012; VDI Heat Atlas PPDS | pure, recommended correlations | to Tc | in `chemicals` (MIT) as coefficients | check each source's terms | refrigerants, common fluids |
| CRC Handbook, "Surface tension of common liquids" | pure | 25 °C plus coefficients | book / online (library) | proprietary | few |
| ITRC PFAS physical and chemical properties table (Aug 2025) | no surface tension; CMC sheet | per value | free Excel (pfas-1.itrcweb.org) | ITRC disclaimer; values cite primary sources | 66 PFAS (density, mp, bp, CMC, pKa, S, VP, KH, Koc) |
| PFAS surfactant literature (below) | aqueous γ(c), γ_CMC, CMC | usually 20-25 °C | articles: tables and figures (digitize) | copyright on the articles; data extracted with citation | the main source for PFAS |
| ACD/Labs (ChemSpider), EPI Suite | predicted | 25 °C | web | proprietary / free | all, but parachor-type estimates, not data |

### Notes per source

- **PubChem / HSDB.** The PUG View heading "Surface Tension" holds HSDB's values, most of them from the CRC
  Handbook (pure liquids at 25 °C: hexane 17.89, ethanol 21.97 mN/m) and the rest from toxicology reviews.
  There is no field that says whether a value is for a pure liquid or a solution; that is in the text, the
  reference, or nowhere. `pubchem enrich --experimental` keeps every value with its source and reference.
  It also collects the density, melting/boiling points and vapor pressure needed for the parachor and for
  telling the phase. PubChem's
  annotation index (`pug_view/annotations/heading/Surface%20Tension`) would list every compound that has a
  surface-tension annotation in a few requests (PUBCHEM-1).
- **EPA CompTox / CTX API.** The Dashboard shows TEST's predicted surface tension (dyn/cm, 25 °C; TEST 5.1 also
  predicts density, viscosity, thermal conductivity, bp, mp, VP, solubility). TEST's models were trained on pure
  liquids, so its PFAS acid predictions are hypothetical pure-liquid values. They make a baseline to compare a
  QuantaBench model against, not training data. The CTX API (api-ccte.epa.gov, free key requested from EPA)
  has `chemical/property/predicted` and `/experimental` endpoints by DTXSID with batch POSTs of up to 1,000
  DTXSIDs. Whether surface tension is among the API's property names should be checked with
  `chemical/property/predicted/name` on the workstation (not reachable from here). OPERA, the Dashboard's
  other model suite, does not predict surface tension.
- **NIST.** WebBook's fluid-property pages give σ(T) of the saturated liquid only for the fluids in its
  equation-of-state set (water, alkanes, refrigerants), which helps for the refrigerant rows. ThermoML is the
  best open bulk source of modern measurements: each data set states the compound, phase, T and method.
  It only covers papers from its contributing journals (J. Chem. Eng. Data, J. Chem. Thermodyn., Fluid
  Phase Equilib., Thermochim. Acta, Int. J. Thermophys.), so older data are missing. TRC's evaluated
  data (WTT/TDE) are the most complete but need a subscription.
- **DIPPR 801, Yaws, Landolt-Börnstein, CRC.** These are the standard evaluated compilations. They are good
  for validating and for filling gaps where a university license exists, but they cannot be redistributed,
  so they cannot go into a public training table. DIPPR's surface-tension QSPR (Knotts, Wilding, Oscarson and
  Rowley, J. Chem. Eng. Data 2001, 46, 1007) is a useful benchmark for the method.
- **Jasper (1972).** About 2,200 pure liquids, each as a linear σ(T) with its temperature range: the classic
  open set for organics. Use it through its coefficients (the `chemicals` package has them) and cite it.
- **ITRC.** The August 2025 workbook has no surface tension. It is still useful twice over: its densities
  (experimental and modelled, with the temperature where stated) feed the parachor, and its CMC sheet
  (18 PFAS with values, mol/L and mg/L) tells where γ_CMC sits on a γ(c) curve.
- **PFAS surfactant literature.** The data for target (2) are in articles, mostly as γ(c) curves:
  - Brusseau (2019, Water Res. 152, 148) compiled about 160 surface-tension data sets from about 65 articles
    for 42 PFAS (PFCAs C2-C11, PFSAs C4-C9, polyfluorinated and nonionic PFAS, FTOHs). He fitted them with
    Szyszkowski and found that the interfacial adsorption coefficient scales with molar volume
    (log Ki = 0.021 Vm − 8.56, r² = 0.94). His Table S1 is the best index of primary sources.
  - Brusseau & Van Glubt (2021, Chemosphere 281, 130829) give data in 0.01 M NaCl and synthetic groundwater.
  - High-purity homologous series: Lunkenheimer et al. (2015), Tamaki et al. (1989), López-Fontán and
    co-workers (sodium PFCAs).
  - Older surfactant work: Kunieda & Shinoda (1976, J. Phys. Chem.: Krafft points, CMCs and surface tension of
    fluorinated surfactants), Kissa, *Fluorinated Surfactants and Repellents* (2001).
  - Moody & Field (2000, Environ. Sci. Technol. 34, 3864), cited by HSDB.

  Values come out of figures by digitizing them (WebPlotDigitizer, Engauge). Record the counter-ion, the
  electrolyte, T and the CMC with every point.

**Conclusion.**
- **Pure-liquid σ:** the open core is Jasper's compilation, PubChem/HSDB (this branch's client) and NIST
  ThermoML, with refrigerants from NIST WebBook or the Mulero/VDI correlations in `chemicals`. DIPPR or TRC,
  through a university license, serve to validate and fill gaps, but not as training data that would be
  published.
- **PFAS γ_CMC:** curated literature data, starting from Brusseau's 2019 compilation and its primary sources,
  with the ITRC CMC sheet to locate the CMC.
- **Predicted values** (TEST on CompTox, ACD/Labs, EPI Suite) are baselines only, not data.

## Merging with other tables

`pubchem enrich` writes one row per input molecule with the input's own columns first, then `pubchem_*`. Join it
to other tables by CAS RN, normalized the same way on both sides (`chem.pubchem.normalize_cas`: leading zeros
dropped, check digit verified). Then join by `pubchem_inchikey`, which equals QuantaBench's standard InChIKey
for the same structure. For a row whose CAS check failed (`pubchem_match` other than `match`), the structure
key, not the CAS, decides which QuantaBench result it belongs to. The long-form `--records` table carries `id`,
`input_cas` and `pubchem_inchikey` on every record, so experimental values join the same way. The ITRC
workbook has its CAS RN in the "CAS No." column of every sheet, one block of rows per compound (the CAS on the
first row only). Its CMC sheet has a second CAS column for the form measured, which can differ from the
compound's (a salt).

## Modeling plan (QSAR-3)

### Data assembly
1. Run QuantaBench in both phases (`build --phases gas,water`). `paired.csv` then has gas_/water_ descriptors
   and dG_solv.
2. `pubchem enrich RUN --validate --experimental --records experimental.csv`. Drop molecules whose
   `pubchem_match` is `different_structure` or `ambiguous_cas` unless resolved by hand, and check
   `component_of_mixture` rows: a measured value belongs to the mixture or salt, not to the component.
3. Target table for model (1): `condition_class == pure_liquid` records within 20-30 °C, or corrected to 25 °C
   with the compound's own dσ/dT where several temperatures exist (else −0.1 mN/m/K). Add Jasper, ThermoML and
   curated literature values with the same provenance columns. Per compound, take the median of independent
   sources; flag spreads above 2 mN/m; exclude `needs_review` records until checked.
   `quantabench-cli pubchem targets experimental.csv --out st25.csv` does this for the PubChem records
   (docs/ENGINES.md, "One target value per molecule"): it also leaves out estimated values, repairs N/m-for-mN/m
   slips, and prefers compilations (CRC, Riddick, Jasper, DIPPR) over other references and the CHRIS manual.
   Two checks it does not make, and that a selection should: a value in water's range (about 65-73 mN/m) for an
   organic liquid is most likely an aqueous solution, and a value far (> 30 %) from the Sugden parachor estimate
   with a measured density is most likely another quantity or another compound (BACKLOG QSAR-1, batch b02).
4. Target table for model (2): γ_CMC and CMC per PFAS, counter-ion and electrolyte, from the curated
   literature. PubChem's unclassified PFAS values such as PFOA's 15.2 mN/m are candidates for review, not data.

### Baselines
- **Sugden parachor.** σ = (P ρ_l / M)⁴ with P from group contributions (Sugden, Quayle, Mumford-Phillips;
  van Krevelen's group values) and ρ_l measured (ITRC, PubChem) or estimated. Errors are amplified fourfold:
  2 % in P or ρ gives 8 % in σ. The enrichment computes an experimental parachor (`pubchem_parachor`, from a
  pure-liquid σ and a liquid density within 10 K) to check group-contribution parachors against. Every run now
  writes Sugden's parachor (`parachor_sugden`) and two density estimates (`density_packing_g_cm3`,
  `density_girolami_g_cm3`), so the baseline needs no measured density ("Density and polarizability estimates"
  below).
- **Parachor from QuantaBench volumes.** The parachor is close to additive in volume, so fit P ≈ a + b·V_vdW
  (+ c·α) on compounds with known σ and ρ. That replaces group counting with computed volumes, which works for
  any structure, including the branched and ether PFAS that group tables cover poorly.
- **TEST predictions** from CompTox, for the same compounds, as the external baseline.

### Direct QSPR on QuantaBench descriptors
The physics: σ is cohesive energy per unit area, roughly ΔH_vap / V_m^(2/3) (Stefan's rule). Its parts are
dispersion, polarity and hydrogen bonding, minus the low polarizability of fluorinated surfaces. Candidate
descriptors (gas phase unless noted):
- **Size and dispersion:** isotropic polarizability α, the vdW volume and SASA (`vdw_volume_A3_best`,
  `sasa_A2_best`; `cpk_volume_A3_best` with Spartan's radii), α/V (a Lorentz–Lorenz-like density of polarizability, the strongest single dispersion term),
  and the molar volume from ρ when known.
- **Polarity:** dipole moment, μ²/V, and the ESP surface descriptors (σ²_tot, ν, Π, positive/negative
  surface areas: the GIPF family, which describes cohesive properties such as heats of vaporization). Add the
  polar surface area from the ESP surface and TPSA.
- **Hydrogen bonding:** donor/acceptor counts, the most positive and most negative ESP values and charges
  (`charge_min_e`/`charge_max_e`).
- **Fluorination:** the F/C ratio, CF2/CF3 counts, and the fraction of the surface that is fluorinated (from
  the ESP surface).
- **Solvation:** dG_solv (water − gas) and the gas → water shifts of μ and α, mainly for model (2).
- **HOMO/LUMO:** expected to add little; keep them only if validation says so.

Models: start with ridge/PLS on about 10-15 standardized descriptors, to keep them interpretable and the
coefficients physically signed. Then try GPR and gradient boosting, which give uncertainties and catch
nonlinearity. Report against the parachor and TEST baselines in MAE and RMSE (mN/m).

### Dimensionless groups and the reduced target (DESC-3)
Ratios with units (α/V in Å³/Å³ is fine, μ²/V is not) mix the size of a molecule with its cohesion. Written in
the liquid's own scale, v_m = V_m / N_A the volume one molecule takes (the packing molar volume when no density is
measured), α, μ, V and A give pure numbers with a physical reading. Every run writes them (`chem.liquid`); the
Analyze table's **Features** menu makes the same groups as formula columns for older run folders
(`analysis.features`), and **Join...** brings measured σ or ρ in from a CSV by id.

| column | group | reads as |
|---|---|---|
| `lorentz_lorenz_bosque` | (4π/3) α / v_m = (n²−1)/(n²+2) | dispersion: the optical dielectric function of the liquid |
| `refractive_index_bosque` | n from it | within ~0.02 of n_D for hexane, perfluorohexane, benzene |
| `dipole_orientation_ratio` | (4π/3) μ²/(3·4πε₀kT) / v_m | dipole-dipole cohesion against kT (Debye's term) |
| `packing_fraction_vdw` | V_vdW / v_m | how tightly the molecules fill the liquid |
| `sphericity_vdw` | π^⅓ (6V)^⅔ / A | 1 for a sphere, lower for chains |
| `hamaker_lifshitz_zJ`, `sigma_lifshitz_mN_m` | A_H (Israelachvili's Lifshitz approximation), σ = A_H / (24π D₀²) | a no-fit σ from n and ε: dispersion only, off by several mN/m |
| `sigma_scale_kT_mN_m` | kT / v_m^⅔ | the unit of the reduced surface tension σ* = σ / this |

**The reduced surface tension** σ* = σ·v_m^⅔/kT is the surface energy of one molecular cross-section in units of kT,
the quantity Eötvös' rule (σ V_m^⅔ ∝ T_c − T) and corresponding states make dimensionless. A model of σ* (then
σ = σ*·kT/v_m^⅔) no longer has to learn that bigger molecules have bigger cross-sections, only how strongly they
cohere. That is the part a fluorinated liquid changes: a perfluoroalkane's Lorentz-Lorenz function is about 30 %
below that of the alkane with the same carbons (hexane 0.23, perfluorohexane 0.16). The groups and the reduced target are candidates to
test under the splits below, against the 26-descriptor models, not defaults.

The Lorentz-Lorenz group needs an optical-like polarizability: GFN2-xTB's static responses run too high and push
it towards 1 (a diverging n), so it uses Bosque's additive α. The dipole groups need a dipole (xTB properties or
a QM stage); a run without one gets the other groups only.

### Splits, validation and applicability domain
- **Splits by cohort and by class, not only at random.** Leave-cohort-out (QuantaBench's cohort labels,
  `chem.validate.cohort_of`). Leave-class-out (PFCA, PFSA, FTOH, perfluoroalkane, ether PFAS, hydrocarbons,
  alcohols, halogenated). Leave-chain-length-out within a homologous series, to test extrapolation to longer
  chains. Bemis-Murcko scaffolds say little for acyclic PFAS, so use them only for the ring compounds.
  Random 5-fold CV is the optimistic bound and is reported as such.
- **Train on organics plus the liquid PFAS, then test on held-out PFAS.** This shows whether the descriptors
  transfer to fluorinated chemistry or the model only interpolates within a class.
- **Applicability domain:**
  - leverage (Williams plot, h* = 3p/n) and k-nearest-neighbour distance in descriptor space;
  - the F/C ratio and size ranges of the training set;
  - a flag on predictions for compounds that are solid at 25 °C, which are hypothetical supercooled-liquid
    values;
  - ionic species, which have no pure-liquid σ in the model's sense.
- **Model (2)** has a few dozen PFAS. Model log CMC and the surface pressure at the CMC (Π_CMC = γ0 − γ_CMC)
  separately, with a few descriptors (fluorinated tail volume, head group, counter-ion), following Brusseau's
  molar-volume result. Validate leave-one-out within classes.

## Density and polarizability estimates: the options (DESC-2)

Batch b02 showed why the quantum parachor σ = ((a·α + b·μ + c)/V)⁴ failed: the parachor follows the liquid's molar
volume V_m (r 0.998), the computed van der Waals volume V is only 50-66 % of V_m, and the minimal-basis xTB
polarizability tracks size poorly (r 0.69 with the parachor). A parachor model needs a liquid density and a
polarizability that scales with size. These are the options, with the accuracy the literature reports and what
QuantaBench measured (docs/VALIDATION.md, DESC-2: densities on 237 PubChem test liquids and 26 fluorinated liquids;
polarizabilities against Lorentz-Lorenz values from PubChem's n_D and ρ and against ωB97X-D/def2-SVPD).

**Density or molar volume**

| option | what it is | accuracy: literature → here | cost | elements, PFAS | needs, license | in QuantaBench |
|---|---|---|---|---|---|---|
| Girolami (J. Chem. Educ. 1994, 71, 962) | relative atom volumes (H 1, Li-F 2, Na-Cl 4, K-Br 5, Rb-I 7.5), +10 % per OH, COOH, NH, amide, sulfoxide or ring, +20 % per sulfone, +7.5 % per fused ring, at most +30 % | RMS 0.049 g/cm³ on 166 liquids → 5.3 % mean error (test set), 6.4 % for liquids with ≥ 3 F, 13 % for P compounds | none (graph) | H-Bi; fluorine is "a second-row atom", no F-specific term | RDKit; a published rule | `density_girolami_g_cm3`, `molar_volume_girolami_cm3` |
| packing coefficient on the vdW volume (DESC-2) | V_m = N_A V_vdW / k, V_vdW the Bondi volume of the best conformer, k linear in 1/n_heavy, element fractions, H-bond donors, ring and aromatic atoms (14 coefficients, fitted on 831 PubChem liquids) | 2.7 % (237 test liquids); 3.2 % for 26 liquids with ≥ 3 F, 3.8 % for the 11 of them the fit never saw (Girolami 6.8 %), 3.0 % above 1.6 g/cm³ | one grid volume (ms) | H C N O F Si P S Cl Br I; 37 fluorinated liquids in its data | RDKit; in the repository (MIT) | `density_packing_g_cm3`, `molar_volume_packing_cm3` |
| GCVOL (Elbro, Fredenslund, Rasmussen, Ind. Eng. Chem. Res. 1991, 30, 2576; extended by Ihmels & Gmehling 2003, 42, 408) | group volumes with a temperature polynomial | the papers report percent-level errors for the classes they cover | none (graph) | the original has no fluorinated groups; check the 2003 table for CF2/CF3 before relying on it | the papers' tables, typed in and checked | not implemented: the tables could not be checked in this session |
| van Krevelen / Bondi group volumes | additive group V_vdW (Bondi 1964) times a packing coefficient | the same idea as the packing model, with tabulated instead of computed volumes | none | Bondi has CF2/CF3 increments | published tables | not implemented: the computed V_vdW does the same job for any structure |
| ML density models | — | no open, pretrained and independently validated model was found (search 2026-11-11); EPA's TEST (public domain) has a density QSAR whose predictions are on the CompTox Dashboard: use it as an external baseline | — | — | — | — |
| COSMO volume | the cavity volume of a COSMO calculation (MOPAC PM7, openCOSMO-RS), then a packing correlation like the one above | about as good as the packing model once calibrated (it is a different volume, not more information) | seconds (MOPAC) | MOPAC's elements | MOPAC (Apache-2.0) | not wired: the COSMO volume is not parsed into descriptors |
| MD of the bulk liquid | NPT simulation of a box of molecules: OpenMM with OpenFF Sage or GAFF2, or LAMMPS | a few percent for organics in published force-field benchmarks (OpenFF fits liquid densities); fluorocarbons are a known weak spot | about 0.5-2 h per molecule on a GPU | the force field's | OpenMM (MIT), OpenFF (MIT), AmberTools | later (the Simulate track) |

**Polarizability**

| option | what it is | accuracy: literature → here | cost | elements, PFAS | needs, license | in QuantaBench |
|---|---|---|---|---|---|---|
| GFN2-xTB response (tblite) | finite-field SCF response of the minimal valence basis | 0.69 × DFT (0.41-2.31: too large for Br, I), r 0.57; 0.65 × Lorentz-Lorenz | 7 SCFs (ms-s) | H-Rn | tblite (LGPL-3.0) | `tblite_gfn2_polarizability_iso_A3`, now in every run with tblite (before: only when tblite-gfn2 refined) |
| GFN1-xTB response (dxtb, tblite) | the same with GFN1-xTB | 0.59 × DFT (0.35-1.47), r 0.72; 0.56 × Lorentz-Lorenz, r 0.85 | 7 SCFs or one autograd pass | H-Rn | dxtb (Apache-2.0) or tblite | `xtb_polarizability_iso_A3` (now also from tblite where dxtb is not installed) |
| D4 model, EEQ charges (Caldeweyher et al., J. Chem. Phys. 2019, 150, 154122) | sum of atomic reference polarizabilities (TD-DFT) scaled by coordination number and charge | 1.01 × DFT for non-fluorinated, 1.11 × DFT for PFAS (r 0.99); 0.96 × Lorentz-Lorenz, r 0.991 | ms | Z ≤ 86 | PyTorch + tad-dftd4 (Apache-2.0) or the dxtb fork; optional | `polarizability_d4_A3` |
| Bosque & Sales (J. Chem. Inf. Comput. Sci. 2002, 42, 1154) | α = 0.32 + Σ n_Z α_Z from the composition | 2.3 % on 340 solvents, 1.9 % on 86 test solvents → 2.8 % against Lorentz-Lorenz (696 PubChem liquids), 4.2 % against DFT (PFAS 3.4 %, r 0.996), 4.8 % against 27 CCCBDB experimental values (water 18 % off) | none | C H N O S P F Cl Br I | a published table | `polarizability_bosque_A3` |
| Wildman-Crippen molar refractivity → α = 3 MR / (4π N_A) (J. Chem. Inf. Comput. Sci. 1999, 39, 868) | atom-type refractivities fitted to measured molar refractivities | 1.7 % against Lorentz-Lorenz (PubChem; it was fitted to that kind of data), 5.0 % against DFT (PFAS 3.5 %, r 0.998), 3.8 % against CCCBDB | none | typed for H C N O F P S Cl Br I (other elements fall into a generic type: no value) | RDKit (BSD) | `polarizability_crippen_A3` |
| Miller's atomic hybrid polarizabilities (J. Am. Chem. Soc. 1990, 112, 8533) | atom-hybrid components (ahc, ahp) | a few percent for organics | none | C N O S P halogens H by hybridization | a published table | not implemented: the table could not be checked in this session; Bosque-Sales and Crippen cover the same ground |
| Lorentz-Lorenz from n_D and ρ | the experimental molar refraction, α = 3/(4π N_A)·(n²−1)/(n²+2)·M/ρ | an optical (589 nm) value, 2-4 % above the static one | data | any liquid with both | PubChem records | the reference of the DESC-2 analysis, not a descriptor |
| DFT in Psi4 (QM stage) | analytic CPKS for RHF/RKS, range-separated hybrids included (ωB97X-D); finite field for meta-GGAs, VV10 and PCM | hybrids with a diffuse basis are within a few percent of experiment; here ωB97X-D/def2-SVPD 0.95 × Lorentz-Lorenz | 3 s to 8 min per molecule of 3-26 atoms at 8 threads (48 in 48 min, DESC-2) | H-Rn with def2 bases; anions need diffuse functions (def2-SVPD, aug-cc-pVDZ) and, in water, the finite field (PCM) | Psi4 (LGPL-3.0) | `--qm wb97x-d/def2-svpd` → `qm_polarizability_iso_A3` |
| ML polarizability models | — | AIMNet2 gives no α; research models for QM9-size molecules exist, none is packaged and validated for PFAS | — | — | — | — |
| Spartan's EST_POLARIZ | Spartan's empirical estimate | trends only | — | — | Spartan | read by the Spartan import and `validate` |

**What to use.** For the liquid density, the packing model (2.7 %; about 4 % for fluorinated liquids it was not
fitted to); Girolami as a graph-only cross-check. For the polarizability, Bosque-Sales or Crippen for absolute
values (within 3-5 % of experiment and of DFT, r ≥ 0.995, PFAS included), D4 where the elements fall outside them (Si, metals) or
a geometry dependence matters (it reads about 10 % high for perfluorinated chains), and the xTB responses only as
relative descriptors or recalibrated. In the b02 surface-tension fits the choice of α moves the cross-validated
error by about 0.4 mN/m once a volume term is in the model, and the density matters more than the α (below).

### b02 v2: the surface tension with the estimates

`Runs/2026-11-11_b02-st100/analysis/report_v2.md` (script `scratch/batches/analyze_b02_v2.py` on NOVA): the 80
training molecules of b02, 5-fold and leave-class-out cross-validation, the 20 holdout molecules predicted once at
the end. Mean absolute errors in mN/m (σ from 16 to 58 mN/m):

| model | needs a measured density | 5-fold CV | leave-class-out | holdout (once) |
|---|---|---|---|---|
| training mean | - | 6.13 | 6.21 | 6.37 |
| Sugden parachor, measured ρ (no fit) | yes | 1.74 | 1.74 | 2.05 |
| quantum parachor (a·α + b·μ² + e·V + c)/V_m, α Bosque-Sales, measured ρ | yes | 1.65 | 1.79 | 1.39 (19: no Si) |
| volume parachor (a·V + c)/V_m, measured ρ | yes | 2.60 | 2.63 | - |
| Sugden parachor, packing ρ (no fit) | no | 3.06 | 3.06 | 3.86 |
| Sugden recalibrated (a·P + b·μ² + c)/V_m, packing ρ | no | 2.87 | 3.00 | 3.01 |
| quantum parachor, α GFN1-xTB, packing ρ | no | 3.05 | 3.12 | 3.51 |
| Sugden parachor, Girolami ρ (no fit) | no | 6.67 | 6.67 | - |
| first report's best density-free form (ln σ on α/V, μ²/V, ...) | no | 4.65 | - | - |

- **With a measured density** the quantum parachor σ^¼ = (a·α + b·μ² + e·V + c)/V_m (V_m = M/ρ) is as good as
  Sugden's group parachor: the α hardly matters once the volume term is in (Bosque-Sales 1.65, GFN1-xTB 1.83, D4
  2.07 in CV). Without the V term, (a·α + b·μ² + c)/V_m fails with every α (CV 7.7 or worse): the parachor is
  mostly a volume.
- **With no measured density** every input comes from the structure. The packing density brings the best
  density-free model from 4.65 (first report, explore_forms.md) to about 2.9 mN/m; Girolami's density is too rough
  for it (Sugden with Girolami's ρ: 6.7). The error that remains is about four times the density error (2.3 % on
  b02's liquids), so a better density, not a better α, is the next step.

## Which surface tension values to train on (DESC-2)

PubChem's surface-tension records come from three kinds of reference, and `pubchem targets` ranks them
(`pubchem_targets.TIERS`): **tier 1**, evaluated compilations (CRC Handbook, Riddick's *Organic Solvents*, Jasper,
DIPPR, Lange, Yaws, TRC); **tier 2**, other HSDB references (encyclopedias, monographs, dossiers); **tier 3**, the
U.S. Coast Guard's CHRIS manual (rounded values, some estimated). The evidence from b02 and its candidates
(`Runs/2026-11-11_b02-st100/analysis/report_v2.md`):

- **One reference per molecule.** Of 371 PubChem molecules with pure-liquid records at 20-30 °C, only one has values
  from two tiers, so the tiers cannot be compared molecule by molecule from PubChem alone. A second independent
  source per molecule (Jasper, ThermoML) is what would give the per-molecule spread.
- **Across molecules, against the Sugden parachor with the measured density** (no fit), on the b02 candidates:
  tier 1 (67 molecules) 1.9 mN/m mean error and 1 % of values more than 30 % off; tier 2 (74) 2.3 mN/m and 3 %;
  CHRIS (49) 2.4 mN/m and 10 %. CHRIS carries the gross errors (dipropylamine 6.1 mN/m against a Sugden 21.7).
- **In cross-validation** the training set (tier 1; tiers 1 + 2; all three, as b02 did) changes the error on the
  tier-1 values by at most 0.15 mN/m, within the noise of 80 molecules, once the screens below have removed the gross
  errors.
- **The 20 → 25 °C correction** (half of b02's values were measured at 20 °C; the molecule's own dσ/dT where its
  records span ≥ 10 K, else −0.1 mN/m per K) lowers the cross-validated error by about 0.1 mN/m for every model with
  a measured density: keep it.
- **The five gases** b02 left out (methylamine, dimethylamine, ethylamine, chloromethane, bromoethene: they boil
  below the record's temperature) are saturated-liquid values that the Sugden parachor reproduces within 0-3 mN/m;
  adding them changes the cross-validated error by less than 0.1 mN/m. They are not wrong, only not liquids at
  1 atm: leave them out of a 25 °C, 1 atm target set.

**Recommendation.** Train on tiers 1 and 2, after the two screens (a value in water's range, ≥ 65 mN/m, for an
organic liquid; a value more than 30 % from the Sugden estimate with the measured density) and the 20 → 25 °C
correction. Keep CHRIS values as a separate check set, and use one in training only where a class has no other
value. Add Jasper's coefficients next (below), so that most molecules get a second, independent value and a spread.

**Other sources and how to get them.**
- **Jasper 1972** (J. Phys. Chem. Ref. Data 1, 841-1010): the standard evaluated σ(T) compilation, about 2,200 pure
  liquids as linear σ = a − bT with their ranges, perfluoroalkanes and other fluorocarbons included. The PDF is free
  from NIST's JPCRD reprints; the coefficients are in the MIT-licensed `chemicals` package (`chemicals.interface`,
  Jasper's data through Lange's Handbook), with CAS numbers to join on. Cite Jasper with every value.
- **NIST ThermoML**: free bulk archive (data.nist.gov) of the XML data files of J. Chem. Eng. Data, J. Chem.
  Thermodyn., Fluid Phase Equilib., Thermochim. Acta and Int. J. Thermophys. since about 2003; each data set states
  compound, phase, T and method. Parse the XML (for example with choderalab's `thermopyl`) and keep the DOI.
- **DIPPR 801** (AIChE): recommended σ(T) correlations; a subscription, values may not be redistributed. Through a
  university license it can validate a model, but it cannot go into a published training table.

**Note for later.** PFAS surfactants (PFCAs longer than C6, PFSAs, their salts) have no pure-liquid σ at 25 °C: their
target is γ at the CMC of an aqueous solution, model (2) of "Two targets", with the counter-ion and electrolyte
recorded. Neither the tiers nor the parachor models above apply to it.

## Open questions
- Which temperature: 25 °C (TEST, CRC, most PFAS work) or 20 °C (much of Jasper and HSDB)? Recommended: 25 °C,
  with converted values flagged. DESC-2: the correction to 25 °C lowers the cross-validated error of b02's models
  by about 0.1 mN/m.
- For PFAS in water (air-water interfaces, foam fractionation, sorption), is γ(c)/γ_CMC the real target, and the
  pure-liquid σ model a stepping stone that validates the descriptors?
- Is there a university license for DIPPR 801 or TRC Web Thermo Tables (validation only)?
