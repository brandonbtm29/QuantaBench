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
  pure-liquid σ and a liquid density within 10 K) to check group-contribution parachors against.
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

## Open questions
- Which temperature: 25 °C (TEST, CRC, most PFAS work) or 20 °C (much of Jasper and HSDB)? Recommended: 25 °C,
  with converted values flagged.
- For PFAS in water (air-water interfaces, foam fractionation, sorption), is γ(c)/γ_CMC the real target, and the
  pure-liquid σ model a stepping stone that validates the descriptors?
- Is there a university license for DIPPR 801 or TRC Web Thermo Tables (validation only)?
