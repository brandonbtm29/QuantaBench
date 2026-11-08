# Methods guide

What each method, conformer search, functional, basis set and solvent model in QuantaBench is, what it is good
and bad at, and what it costs. The same text is in the app (the info under each pipeline choice, and
Help > Methods guide...) and on the command line (`quantabench-cli methods`, `quantabench-cli methods wb97x-d`).

QuantaBench does not try to reproduce Spartan exactly: its default levels include a dispersion correction, and
several of the Spartan results it is validated against (B3LYP, PM6) have none. The Spartan entries say what those
numbers are. Numbers marked "QuantaBench measured" are from [VALIDATION.md](VALIDATION.md),
[ENGINES.md](ENGINES.md) and [dev/BACKLOG.md](dev/BACKLOG.md); references were checked against Crossref.

This file is generated from `src/quantabench/chem/method_guide.py` by `python scripts/write_methods_doc.py`.
Edit the data there, not this file (a test checks that the two agree).

## Contents

- **Machine-learning potentials:** [AIMNet2](#aimnet2)
- **Tight binding (xTB):** [GFN2-xTB](#gfn2-xtb), [GFN1-xTB](#gfn1-xtb)
- **Semi-empirical (NDDO):** [PM7](#pm7), [PM6-D3H4X](#pm6-d3h4x)
- **Force fields:** [MMFF94s](#mmff94s), [UFF](#uff), [GFN-FF](#gfn-ff)
- **Conformer search:** [RDKit ETKDGv3 + MMFF94s](#rdkit-etkdg), [CREST iMTD-GC](#crest), [CREST level:
  GFN2-xTB](#crest-gfn2), [CREST level: GFN-FF](#crest-gfnff), [CREST level: GFN2-xTB //
  GFN-FF](#crest-gfn2--gfnff), [CREST level: GFN1-xTB](#crest-gfn1)
- **DFT functionals and HF:** [DFT and functional names](#dft), [ωB97X-D](#wb97x-d),
  [B3LYP-D3(BJ)](#b3lyp-d3bj), [B3LYP (no dispersion)](#b3lyp), [PBE0-D3(BJ)](#pbe0-d3bj), [r2SCAN](#r2scan),
  [M06-2X](#m06-2x), [Hartree-Fock (HF)](#hf), [ωB97M-D3(BJ) / ωB97M-V](#wb97m)
- **Basis sets:** [Basis sets (what the names mean)](#basis-sets), [def2-SVP](#def2-svp),
  [def2-SVPD](#def2-svpd), [def2-TZVP](#def2-tzvp), [6-31G\*](#6-31gs), [6-31+G\*](#6-31pgs),
  [6-311+G\*\*](#6-311pgss), [aug-cc-pVDZ](#aug-cc-pvdz)
- **Implicit solvation:** [Implicit solvation](#solvation), [Gas phase](#gas-phase), [ALPB](#alpb),
  [GBSA](#gbsa), [COSMO](#cosmo), [PCM (IEF-PCM, C-PCM)](#pcm), [SMD](#smd)
- **Optimizers:** [L-BFGS (ASE)](#lbfgs), [FIRE (ASE)](#fire), [Engine's own optimizer](#native-optimizer)
- **Spartan's levels:** [Spartan B3LYP/6-311+G\*\* (C-PCM water)](#spartan-b3lyp), [Spartan
  ωB97X-D/6-31G\*](#spartan-wb97x-d), [Spartan PM6](#spartan-pm6)
- **Concepts:** [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [Single point vs
  optimization](#single-point), [Boltzmann weighting](#boltzmann), [HOMO and LUMO energies](#homo-lumo), [Auto
  (best available)](#auto)

## Machine-learning potentials

<a id="aimnet2"></a>

### AIMNet2

> Neural network trained to reproduce ωB97M-D3(BJ)/def2-TZVPP DFT; 14 elements, closed shells, gas phase.

A machine-learned interatomic potential: a neural network maps the atoms' positions and elements and the total
charge to an energy, forces and atomic charges. It was trained on about 20 million hybrid-DFT calculations
([ωB97M](#wb97m)/def2-TZVPP) of neutral and charged molecules. Long-range electrostatics and D3(BJ) dispersion
are explicit physics terms added to the network's short-range energy.

- **Good for:** Conformer energies and geometries of organic molecules and ions made of H, B, C, N, O, F, Si,
  P, S, Cl, As, Se, Br and I, at a small fraction of the DFT cost. The first choice of `auto` when it is
  installed. The best cheap gas-phase conformer ranker QuantaBench tested on neutral PFAS: its lowest
  conformer was within 4 kJ/mol of ωB97X-D/6-31G\*'s in 45 of 53 Essential 76 molecules (GFN2-xTB 36, a random
  pick 25; docs/VALIDATION.md, RANK-1).
- **Watch out:** Only those 14 elements, no transition metals, closed shells only (radicals need AIMNet2-NSE,
  which QuantaBench does not use). Its developers validated it up to about 100 heavy atoms. No implicit
  solvent in QuantaBench, so a water run refines in the gas phase (with a note). No orbitals, so no HOMO/LUMO;
  the dipole comes from its predicted atomic charges. It ranks the conformers of PFAS ions in water only
  moderately, even with a solvation term from GFN2-xTB/CPCM added (median ρ 0.64 against ωB97X-D/def2-SVP in
  PCM over 17 ions, 0.50 on the 9 hardest; RANK-1), and leans toward extended conformers.
- **Cost:** Fastest on a GPU, which refines all conformers of a molecule in one batch: QuantaBench measured 8
  PFOA conformers in 5.7 s on an RX 9060 XT and 59.8 s on the CPU (docs/PLATFORMS.md).
- **Dispersion:** Yes: D3 with Becke-Johnson damping (ωB97M parameters), added to the network energy.
- **Solvation:** None (gas phase).
- **In QuantaBench:**
  - `aimnet2`: aimnet package, in process. NVIDIA GPUs and the CPU through NVIDIA Warp; AMD (ROCm) and Apple
    GPUs through QuantaBench's Warp-free path (`chem/aimnet_torch.py`), the same energies to \< 1e-5 eV.
    Batched refinement on a GPU.
- **See also:** [ωB97M-D3(BJ) / ωB97M-V](#wb97m), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion),
  [Auto (best available)](#auto)
- **References:**
  - Anstine, Zubatyuk, Isayev. AIMNet2: a neural network potential to meet your neutral, charged, organic, and
    elemental-organic needs. Chem. Sci. 16, 10228-10244 (2025).
    [doi:10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
- **Also found as:** `aimnet`, `aimnet 2`

## Tight binding (xTB)

<a id="gfn2-xtb"></a>

### GFN2-xTB

> Grimme's extended tight-binding method, a fast approximation to DFT: good geometries and conformers, H to
> Rn.

Tight binding is a semi-empirical simplification of DFT: a minimal valence basis, element parameters and
approximate integrals. GFN stands for the targets of the fit: geometries, frequencies and noncovalent
interactions. GFN2 adds atomic dipole and quadrupole electrostatics to GFN1's atomic charges and computes the
charge-dependent [D4 dispersion](#dispersion) self-consistently. It has parameters for every element up to
radon (Z = 86).

- **Good for:** Geometries and conformer ranking of organic molecules, ions and organometallics, in the gas
  phase or in [ALPB](#alpb) solvent, at low cost. Descriptors (dipole, charges, polarizability) for trends
  across molecules. QuantaBench's default refinement level for the PFAS validation.
- **Watch out:** Orbital energies are not what it was fitted to: treat HOMO, LUMO and the gap as trend
  descriptors ([HOMO and LUMO energies](#homo-lumo)). QuantaBench measured, at the same geometries: gas-phase
  dipoles 0.81 D MAE from ωB97X-D/6-31G\* (0.76 D higher on average, 63 PFAS), and water dipoles 0.52 D MAE
  from Spartan's B3LYP/C-PCM (61 ions; docs/VALIDATION.md, VAL-3). In the gas phase a multiply charged anion
  can have an unbound HOMO, and its polarizability can then diverge (QuantaBench discards implausible values
  with a note). Do not rank the conformers of PFAS ions with it, in any solvent model: against
  ωB97X-D/def2-SVP in PCM its lowest conformer was within 4 kJ/mol of DFT's in 4 of 17 ions with ALPB (5 with
  CPCM, 4 in the gas phase), no better than a random pick, and its median Spearman ρ was −0.13
  (docs/VALIDATION.md, RANK-1). It puts conformers with short anionic O···F contacts (\< 2.6 Å) 10-20 kJ/mol
  per contact too low, compactness held fixed. Its 6 kcal/mol window would have dropped DFT's lowest in 6 of
  the 17. Use it for geometries; rank ions' conformers with the QM stage in PCM (RANK-1 has a cheap
  alternative). For neutral PFAS in the gas phase its picks were within 4 kJ/mol of ωB97X-D/6-31G\*'s in 36 of
  53 molecules.
- **Cost:** Milliseconds to a fraction of a second per energy and gradient for PFAS-size molecules on one
  core. QuantaBench measured the whole pipeline for 24 PFAS on BTM NOVA: 126 s with tblite GFN2, against 76 s
  with AIMNet2 and 34 s with MMFF94s alone (docs/dev/BACKLOG.md, PERF-1).
- **Dispersion:** Yes: D4, charge-dependent and self-consistent.
- **Solvation:** ALPB and GBSA (tblite, the xtb program, CREST); CPCM (tblite). None in dxtb.
- **In QuantaBench:**
  - `gfn2-xtb`: dxtb (PyTorch, in process): autograd forces and Hessians, batched refinement. It can run on a
    GPU but is 5-8x slower there than on the CPU (dxtb 0.4), so it stays on the CPU by default. Gas phase
    only. Needs dxtb's libcint backend (the dxtb fork on Windows and macOS).
  - `tblite-gfn2`: tblite (the reference library, in process, CPU): ALPB, GBSA or CPCM solvent; dipole,
    HOMO/LUMO, finite-field polarizability and the solvation energy. Matches the xtb program to 4e-9 Eh.
  - `xtb-gfn2`: the xtb program (one external run per calculation): ALPB or GBSA; its own optimizer, 4-10x
    faster than ASE stepping xtb; its polarizability is the D4 model's α(0), not the SCF response.
- **See also:** [GFN1-xTB](#gfn1-xtb), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion),
  [ALPB](#alpb), [HOMO and LUMO energies](#homo-lumo), [CREST level: GFN2-xTB](#crest-gfn2)
- **References:**
  - Bannwarth, Ehlert, Grimme. GFN2-xTB, an accurate and broadly parametrized self-consistent tight-binding
    quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. J.
    Chem. Theory Comput. 15, 1652-1671 (2019).
    [doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
  - Bannwarth, Caldeweyher, Ehlert, Hansen, Pracht, Seibert, Spicher, Grimme. Extended tight-binding quantum
    chemistry methods. WIREs Comput. Mol. Sci. 11, e1493 (2021).
    [doi:10.1002/wcms.1493](https://doi.org/10.1002/wcms.1493)
  - Friede, Hölzer, Ehlert, Grimme. dxtb, an efficient and fully differentiable framework for extended
    tight-binding. J. Chem. Phys. 161, 062501 (2024).
    [doi:10.1063/5.0216715](https://doi.org/10.1063/5.0216715)
- **Also found as:** `gfn2`, `gfn2 xtb`, `xtb`, `gfn2-xtb (dxtb)`, `gfn2-xtb (tblite)`, `gfn2-xtb (xtb
  program)`

<a id="gfn1-xtb"></a>

### GFN1-xTB

> The first GFN tight-binding method: like GFN2 with atomic charges only, D3(BJ) dispersion and a halogen
> term.

GFN1-xTB (2017) is the first of the GFN tight-binding methods. Its electrostatics use atomic charges only (no
atomic dipoles or quadrupoles), dispersion is [D3 with Becke-Johnson damping](#dispersion), and a halogen-bond
correction is added. It has parameters for H to Rn (Z = 1-86).

- **Good for:** A cross-check of GFN2-xTB results. QuantaBench's xTB properties step (the `xtb_*` charges,
  dipole and polarizability) uses it. dxtb runs it without the libcint backend, on the CPU or a GPU. With
  tblite's CPCM (ddX) its solvation energy differences between conformers of PFAS ions follow IEF-PCM's at
  ωB97X-D/def2-SVP to 3.9 kJ/mol MAE, closer than GFN2 with CPCM (9.7) or ALPB (20). Added to g-xTB's
  gas-phase energy, that term gave the cheap ranker closest to both DFT references tested (docs/VALIDATION.md,
  RANK-1).
- **Watch out:** The same caveats as GFN2-xTB for orbital energies ([HOMO and LUMO energies](#homo-lumo)) and
  for gas-phase anions. Its simpler electrostatics describe anisotropic contacts (hydrogen and halogen bonds)
  less well than GFN2's multipoles; GFN1 corrects halogen bonds with an extra term.
- **Cost:** About the same as GFN2-xTB.
- **Dispersion:** Yes: D3 with Becke-Johnson damping.
- **Solvation:** ALPB and GBSA (tblite, the xtb program, CREST); CPCM (tblite). None in dxtb.
- **In QuantaBench:**
  - `gfn1-xtb`: dxtb (PyTorch, in process): CPU or GPU, batched refinement, gas phase. Agrees with tblite's
    GFN1 to 5e-5 e in charges and 1e-4 D in dipoles (docs/ENGINES.md).
  - `tblite-gfn1`: tblite (in process, CPU): ALPB, GBSA or CPCM solvent; dipole, HOMO/LUMO and finite-field
    polarizability.
  - `xtb-gfn1`: the xtb program (external): ALPB or GBSA, its own optimizer.
- **See also:** [GFN2-xTB](#gfn2-xtb), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [CREST
  level: GFN1-xTB](#crest-gfn1)
- **References:**
  - Grimme, Bannwarth, Shushkov. A robust and accurate tight-binding quantum chemical method for structures,
    vibrational frequencies, and noncovalent interactions of large molecular systems parametrized for all
    spd-block elements (Z = 1-86). J. Chem. Theory Comput. 13, 1989-2009 (2017).
    [doi:10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)
  - Bannwarth, Caldeweyher, Ehlert, Hansen, Pracht, Seibert, Spicher, Grimme. Extended tight-binding quantum
    chemistry methods. WIREs Comput. Mol. Sci. 11, e1493 (2021).
    [doi:10.1002/wcms.1493](https://doi.org/10.1002/wcms.1493)
- **Also found as:** `gfn1`, `gfn1 xtb`, `gfn-xtb`

## Semi-empirical (NDDO)

<a id="pm7"></a>

### PM7

> Stewart's NDDO semi-empirical method (MOPAC), fitted to experiment; dispersion and H-bond terms built in.

NDDO methods (neglect of diatomic differential overlap) solve a minimal-basis Hartree-Fock problem in which
most two-electron integrals are dropped and the rest come from fitted parameters. PM7 (2013) was fitted to
experimental and high-level reference data, crystal structures included, with dispersion and hydrogen-bond
terms in the method. MOPAC reports the energy as a heat of formation ΔHf.

- **Good for:** Heats of formation, dipoles and polarizabilities (MOPAC POLAR) of organic molecules; COSMO
  areas and volumes as size descriptors; a second semi-empirical opinion next to xTB. For simple gas-phase
  organic molecules Stewart reports average ΔHf errors about 10 % below PM6's.
- **Watch out:** The energy is a heat of formation: compare it only within PM7. [COSMO](#cosmo) in MOPAC is
  not smooth, so optimizations in water can stop short of the force threshold and COSMO frequencies are
  approximate. Elements: H to La and Lu to Bi. It ranks the conformers of PFAS ions as badly as GFN2-xTB does
  (median ρ against ωB97X-D/PCM over 17 ions −0.38 gas, −0.05 COSMO) and overbinds short anionic O···F
  contacts about as much (docs/VALIDATION.md, RANK-1).
- **Cost:** One MOPAC run (an external program) per energy and gradient. With Optimizer = engine's own, a
  whole optimization is one run, 3-15x faster than ASE stepping MOPAC (docs/ENGINES.md).
- **Dispersion:** Yes: a damped pairwise dispersion term and a hydrogen-bond term, part of the method.
- **Solvation:** COSMO (from the dielectric constant).
- **In QuantaBench:**
  - `mopac-pm7`: MOPAC 22 or later (open source, external): COSMO solvent; heat of formation, dipole,
    HOMO/LUMO, ionization energy, POLAR polarizability (closed shells).
- **See also:** [PM6-D3H4X](#pm6-d3h4x), [COSMO](#cosmo), [Spartan PM6](#spartan-pm6)
- **References:**
  - Stewart. Optimization of parameters for semiempirical methods VI: more modifications to the NDDO
    approximations and re-optimization of parameters. J. Mol. Model. 19, 1-32 (2013).
    [doi:10.1007/s00894-012-1667-x](https://doi.org/10.1007/s00894-012-1667-x)
- **Also found as:** `pm7 (mopac)`

<a id="pm6-d3h4x"></a>

### PM6-D3H4X

> PM6 plus Grimme's D3 dispersion and Řezáč-Hobza hydrogen-bond (H4) and halogen (X) corrections (MOPAC).

PM6 (2007) is the NDDO method before PM7, parametrized for 70 elements. Plain PM6 describes dispersion and
hydrogen bonds poorly. D3H4X adds Grimme's D3 dispersion, the H4 hydrogen-bond correction and a correction for
halogen-oxygen and halogen-nitrogen contacts. QuantaBench's `mopac-pm6` engine always runs PM6-D3H4X.

- **Good for:** Noncovalent contacts (stacking, hydrogen and halogen bonds) at semi-empirical cost.
- **Watch out:** Not the same as Spartan's PM6, which is plain PM6 ([Spartan PM6](#spartan-pm6)): geometries
  and conformer energies of flexible molecules can differ. The energy is a heat of formation; COSMO caveats as
  for [PM7](#pm7). In the gas phase it overbinds short anionic O···F contacts as GFN2-xTB does; with COSMO it
  ranked PFAS ions' conformers better (median ρ 0.52 against ωB97X-D/PCM over 17 ions) but still picked one
  more than 4 kJ/mol above DFT's lowest in 4 of 17 (docs/VALIDATION.md, RANK-1).
- **Cost:** One MOPAC run (an external program) per energy and gradient. With Optimizer = engine's own, a
  whole optimization is one run, 3-15x faster than ASE stepping MOPAC (docs/ENGINES.md).
- **Dispersion:** Yes: D3, plus the H4 hydrogen-bond and X halogen corrections.
- **Solvation:** COSMO (from the dielectric constant).
- **In QuantaBench:**
  - `mopac-pm6`: MOPAC 22 or later (open source, external): COSMO solvent; the same properties as PM7.
- **See also:** [PM7](#pm7), [Spartan PM6](#spartan-pm6), [Dispersion corrections (D3, D3(BJ), D4,
  -D)](#dispersion)
- **References:**
  - Stewart. Optimization of parameters for semiempirical methods V: modification of NDDO approximations and
    application to 70 elements. J. Mol. Model. 13, 1173-1213 (2007).
    [doi:10.1007/s00894-007-0233-4](https://doi.org/10.1007/s00894-007-0233-4)
  - Řezáč, Hobza. Advanced corrections of hydrogen bonding and dispersion for semiempirical quantum mechanical
    methods. J. Chem. Theory Comput. 8, 141-151 (2012).
    [doi:10.1021/ct200751e](https://doi.org/10.1021/ct200751e)
  - Brahmkshatriya, Dobeš, Fanfrlík, Řezáč, Paruch, Bronowska, Lepšík, Hobza. Quantum mechanical scoring:
    structural and energetic insights into cyclin-dependent kinase 2 inhibition by
    pyrazolo\[1,5-a\]pyrimidines. Curr. Comput.-Aided Drug Des. 9, 118-129 (2013).
    [doi:10.2174/1573409911309010011](https://doi.org/10.2174/1573409911309010011)
- **Also found as:** `pm6`, `pm6-d3h4`, `pm6 d3h4x (mopac)`

## Force fields

<a id="mmff94s"></a>

### MMFF94s

> Merck molecular force field (static variant): classical terms, no electrons; milliseconds, used for
> conformers.

A force field computes the energy from bond stretches, angle bends, torsions, van der Waals contacts and fixed
atomic charges, with no electrons. MMFF94 was parametrized on quantum-chemical data for organic and drug-like
molecules; the 's' variant makes delocalized trigonal nitrogens (as in anilines) planar, which suits energy
minimization. RDKit's MMFF94s minimizes every embedded conformer, and the method `mmff` stops there.

- **Good for:** Fast pre-optimization and a first ranking of many conformers; large batches where only
  geometries and 2D/3D shape descriptors are needed.
- **Watch out:** Fixed charges: no polarization, no solvent, no electronic properties. Its conformer ranking
  is rough and in the gas phase, which is why QuantaBench refines the lowest 20 by default: as single points
  at fixed geometries its relative energies were 51 kJ/mol off ωB97X-D/PCM's on average for PFAS ions and 8
  kJ/mol off ωB97X-D/6-31G\*'s for neutral PFAS (docs/VALIDATION.md, RANK-1). Molecules it has no parameters
  for (Sn, Hg, B, hypervalent iodine, radicals) go to [UFF](#uff).
- **Cost:** Milliseconds per conformer. QuantaBench measured 24 PFAS through the MMFF94s-only pipeline in 34 s
  on BTM NOVA (docs/dev/BACKLOG.md, PERF-1).
- **Dispersion:** A buffered 14-7 van der Waals term (part of the force field, not a DFT-style correction).
- **Solvation:** None.
- **In QuantaBench:**
  - `mmff`: RDKit, in process, CPU threads; UFF for molecules MMFF94s has no parameters for.
- **See also:** [UFF](#uff), [RDKit ETKDGv3 + MMFF94s](#rdkit-etkdg)
- **References:**
  - Halgren. Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94.
    J. Comput. Chem. 17, 490-519 (1996).
    [doi:10.1002/(SICI)1096-987X(199604)17:5/6\<490::AID-JCC1\>3.0.CO;2-P](https://doi.org/10.1002/%28SICI%291096-987X%28199604%2917%3A5/6%3C490%3A%3AAID-JCC1%3E3.0.CO%3B2-P)
  - Halgren. MMFF VI. MMFF94s option for energy minimization studies. J. Comput. Chem. 20, 720-729 (1999).
    [doi:10.1002/(SICI)1096-987X(199905)20:7\<720::AID-JCC7\>3.0.CO;2-X](https://doi.org/10.1002/%28SICI%291096-987X%28199905%2920%3A7%3C720%3A%3AAID-JCC7%3E3.0.CO%3B2-X)
- **Also found as:** `mmff`, `mmff94`, `mmff94s (force field)`

<a id="uff"></a>

### UFF

> Universal force field: rule-based parameters for the whole periodic table; the fallback when MMFF94s has
> none.

UFF builds its parameters from element and hybridization rules, so it covers the whole periodic table.
QuantaBench (through RDKit) uses it for the conformer search when MMFF94s lacks parameters for a molecule, and
says so in the notes and the `forcefield` column.

- **Good for:** Starting geometries for organometallics, tin and mercury compounds and other unusual valences.
- **Watch out:** Less accurate than MMFF94s for organic molecules: treat its conformer energies only as a
  pre-sort before the refinement.
- **Cost:** Milliseconds per conformer.
- **Dispersion:** A Lennard-Jones van der Waals term.
- **Solvation:** None.
- **See also:** [MMFF94s](#mmff94s)
- **References:**
  - Rappé, Casewit, Colwell, Goddard, Skiff. UFF, a full periodic table force field for molecular mechanics
    and molecular dynamics simulations. J. Am. Chem. Soc. 114, 10024-10035 (1992).
    [doi:10.1021/ja00051a040](https://doi.org/10.1021/ja00051a040)
- **Also found as:** `universal force field`

<a id="gfn-ff"></a>

### GFN-FF

> Grimme's generic force field (H to Rn) that sets itself up from the structure; much faster than xTB.

GFN-FF builds its own force field from the coordinates and elements: bond topology, bond orders and EEQ atomic
charges come from the structure, so no atom types are needed. Dispersion is a simplified D4. On the
conformational subsets of the GMTKN55 benchmark it has a mean absolute deviation of 1.53 kcal/mol, which its
authors report as on par with the GFN-xTB methods.

- **Good for:** Large or unusual molecules, molecules with metals, and fast conformer searches ([CREST level:
  GFN-FF](#crest-gfnff)).
- **Watch out:** The bond topology is fixed when it is set up: bonds cannot form or break. The dipole comes
  from the EEQ charges, not from an electron density, and there are no orbitals. Its conformer energies are a
  search level, not a ranking: for neutral PFAS its picks were no better than random against ωB97X-D/6-31G\*
  (within 4 kJ/mol in 22 of 53), and its errors for PFAS anions reached 100 kJ/mol (docs/VALIDATION.md,
  RANK-1).
- **Cost:** Far below xTB per step. CREST with GFN-FF searched 20 PFAS more than 6x faster than CREST with
  GFN2 on the same machine (QuantaBench measured, docs/ENGINES.md).
- **Dispersion:** Yes: a simplified D4.
- **Solvation:** ALPB and GBSA (the xtb program, CREST).
- **In QuantaBench:**
  - `gfnff`: the xtb program (external): ALPB or GBSA, its own optimizer; EEQ charges and the charge dipole as
    properties.
- **See also:** [CREST level: GFN-FF](#crest-gfnff), [GFN2-xTB](#gfn2-xtb)
- **References:**
  - Spicher, Grimme. Robust atomistic modeling of materials, organometallic, and biochemical systems. Angew.
    Chem. Int. Ed. 59, 15665-15673 (2020).
    [doi:10.1002/anie.202004239](https://doi.org/10.1002/anie.202004239)
- **Also found as:** `gfnff`, `gff`, `gfn-ff (xtb program)`

## Conformer search

<a id="rdkit-etkdg"></a>

### RDKit ETKDGv3 + MMFF94s

> Distance-geometry embedding with experimental torsion preferences, then MMFF94s minimization; seconds per
> molecule.

ETKDG builds 3D structures by distance geometry: random interatomic distances within bond-length and angle
bounds, embedded in 3D and corrected with torsion-angle preferences from crystal structures and simple rules
(flat aromatic rings, linear triple bonds). Version 3 improves small rings and macrocycles. QuantaBench embeds
up to Max conformers structures (50 by default), minimizes each with [MMFF94s](#mmff94s), and prunes by the
energy window and an RMSD duplicate test.

- **Good for:** The default search: fast, reproducible with the random seed, and it needs no external program.
- **Watch out:** The pool is ranked by gas-phase MMFF94s and only the lowest go on to the refinement, so
  conformers that a quantum method in water prefers can be lost there. In QuantaBench's comparison with
  Spartan (refining the 5 lowest), Spartan's pool held a conformer GFN2/ALPB puts below QuantaBench's lowest
  for 49 of 61 ions (docs/VALIDATION.md, VAL-3). Raise Refine lowest for flexible ions, or use
  [CREST](#crest).
- **Cost:** Seconds per molecule for PFAS. A cyclic peptide of 199 atoms took about 1.4 s per conformer on one
  core (docs/ENGINES.md).
- **Dispersion:** Only MMFF94s's van der Waals term.
- **Solvation:** None: gas phase, the same pool in every phase.
- **See also:** [MMFF94s](#mmff94s), [CREST iMTD-GC](#crest), [Boltzmann weighting](#boltzmann)
- **References:**
  - Riniker, Landrum. Better informed distance geometry: using what we know to improve conformation
    generation. J. Chem. Inf. Model. 55, 2562-2574 (2015).
    [doi:10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)
  - Wang, Witek, Landrum, Riniker. Improving conformer generation for small rings and macrocycles based on
    distance geometry and experimental torsional-angle preferences. J. Chem. Inf. Model. 60, 2044-2058 (2020).
    [doi:10.1021/acs.jcim.0c00025](https://doi.org/10.1021/acs.jcim.0c00025)
- **Also found as:** `rdkit`, `etkdg`, `etkdgv3`, `etkdg v3`

<a id="crest"></a>

### CREST iMTD-GC

> Grimme's metadynamics conformer search on xTB or GFN-FF: finds conformers ETKDG misses; minutes to hours.

CREST explores conformers with iterative metadynamics (iMTD): molecular dynamics with a bias that pushes the
structure away from conformers it has already visited, measured by RMSD. Low structures are optimized, and a
genetic crossover step (GC) combines parts of different conformers. Energies come from the chosen CREST level,
in [ALPB](#alpb) solvent when one is set. Quick mode (`--quick`, on by default here) uses reduced settings for
a cruder ensemble; turn it off for CREST's full search.

- **Good for:** Flexible molecules and ions, especially in water, where a gas-phase force-field pool ranks
  poorly; ensembles for Boltzmann averages.
- **Watch out:** Needs the crest program (Linux, the fork build on macOS, WSL on Windows). Run time grows
  quickly with size and flexibility. CREST can find tautomers or broken bonds: QuantaBench drops conformers
  whose bonds differ from the input. If CREST fails, the molecule gets RDKit's conformers (when that box is
  checked) and a note.
- **Cost:** GFN-FF `--quick`: ethanol about 2 s. GFN2 `--quick`: PFBA about 2 min on 2 cores, a C8
  perfluoroalkyl acid tens of minutes (chem/crest.py).
- **Dispersion:** That of the CREST level: D4 (GFN2-xTB, GFN-FF) or D3(BJ) (GFN1-xTB).
- **Solvation:** ALPB.
- **See also:** [CREST level: GFN2-xTB](#crest-gfn2), [CREST level: GFN-FF](#crest-gfnff), [CREST level:
  GFN2-xTB // GFN-FF](#crest-gfn2--gfnff), [CREST level: GFN1-xTB](#crest-gfn1), [RDKit ETKDGv3 +
  MMFF94s](#rdkit-etkdg)
- **References:**
  - Pracht, Bohle, Grimme. Automated exploration of the low-energy chemical space with fast quantum chemical
    methods. Phys. Chem. Chem. Phys. 22, 7169-7192 (2020).
    [doi:10.1039/C9CP06869D](https://doi.org/10.1039/C9CP06869D)
  - Pracht et al.. CREST, a program for the exploration of low-energy molecular chemical space. J. Chem. Phys.
    160, 114110 (2024). [doi:10.1063/5.0197592](https://doi.org/10.1063/5.0197592)
- **Also found as:** `crest imtd-gc`, `imtd-gc`, `imtd`, `crest quick`, `--quick`

<a id="crest-gfn2"></a>

### CREST level: GFN2-xTB

> The whole CREST search at GFN2-xTB: CREST's default and most accurate level, and the slowest.

Metadynamics, optimizations and the final ranking all use [GFN2-xTB](#gfn2-xtb), in ALPB when a solvent is
set.

- **Good for:** Final conformer ensembles when there is time, ions in water.
- **Watch out:** Large flexible PFAS take hours: on the M3 Max (6 lanes x 2 threads) GFN2 took 20,034 s for 16
  of 20 PFAS in the gas phase, and the 4 largest hit the 3 h limit per molecule (QuantaBench measured,
  docs/ENGINES.md).
- **Cost:** Minutes for small acids, hours for large flexible ones (above).
- **Dispersion:** Yes: D4.
- **Solvation:** ALPB.
- **See also:** [CREST iMTD-GC](#crest), [GFN2-xTB](#gfn2-xtb)
- **References:**
  - Pracht, Bohle, Grimme. Automated exploration of the low-energy chemical space with fast quantum chemical
    methods. Phys. Chem. Chem. Phys. 22, 7169-7192 (2020).
    [doi:10.1039/C9CP06869D](https://doi.org/10.1039/C9CP06869D)
  - Bannwarth, Ehlert, Grimme. GFN2-xTB, an accurate and broadly parametrized self-consistent tight-binding
    quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. J.
    Chem. Theory Comput. 15, 1652-1671 (2019).
    [doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- **Also found as:** `crest gfn2`, `crest --gfn2`

<a id="crest-gfnff"></a>

### CREST level: GFN-FF

> The whole search with the GFN-FF force field: more than 6x faster than GFN2; ranking at force-field level.

Metadynamics, optimizations and the ranking use [GFN-FF](#gfn-ff). The Refinement stage then re-optimizes the
lowest conformers with the refinement method.

- **Good for:** Large molecules and big batches; a first ensemble that the refinement re-ranks.
- **Watch out:** The ranking is at force-field level, so refine with xTB or better (the pipeline does). Bonds
  are fixed by GFN-FF's topology.
- **Cost:** QuantaBench measured 20 PFAS in the gas phase: 3,108 s on 6 lanes x 2 threads and 773 s on 3 x 4
  threads (M3 Max), 1,247 s on BTM NOVA's WSL (20 x 2) (docs/ENGINES.md).
- **Dispersion:** Yes: a simplified D4.
- **Solvation:** ALPB.
- **See also:** [CREST iMTD-GC](#crest), [GFN-FF](#gfn-ff)
- **References:**
  - Pracht, Bohle, Grimme. Automated exploration of the low-energy chemical space with fast quantum chemical
    methods. Phys. Chem. Chem. Phys. 22, 7169-7192 (2020).
    [doi:10.1039/C9CP06869D](https://doi.org/10.1039/C9CP06869D)
  - Spicher, Grimme. Robust atomistic modeling of materials, organometallic, and biochemical systems. Angew.
    Chem. Int. Ed. 59, 15665-15673 (2020).
    [doi:10.1002/anie.202004239](https://doi.org/10.1002/anie.202004239)
- **Also found as:** `crest gfnff`, `crest gff`, `crest --gfnff`

<a id="crest-gfn2--gfnff"></a>

### CREST level: GFN2-xTB // GFN-FF

> Samples and optimizes with GFN-FF, ranks with GFN2-xTB single points; needs the xtb program.

'A//B' means energies at A on geometries from B. CREST runs the metadynamics and optimizations with
[GFN-FF](#gfn-ff) and evaluates the energies and sorts the ensemble with [GFN2-xTB](#gfn2-xtb). Its accuracy
lies between the two levels, because the geometries keep GFN-FF's biases (CREST's developers, crest-lab/crest
issue 72).

- **Good for:** Big searches that should be ranked at a quantum level.
- **Watch out:** CREST 3.0.2 runs this level only in `--legacy` mode, which calls the xtb program for every
  calculation, so it needs xtb and is slower than CREST's built-in GFN-FF.
- **Cost:** Between GFN-FF and GFN2-xTB.
- **Dispersion:** Yes: D4 (both levels).
- **Solvation:** ALPB.
- **See also:** [CREST iMTD-GC](#crest), [Single point vs optimization](#single-point), [GFN-FF](#gfn-ff),
  [GFN2-xTB](#gfn2-xtb)
- **References:**
  - Pracht, Bohle, Grimme. Automated exploration of the low-energy chemical space with fast quantum chemical
    methods. Phys. Chem. Chem. Phys. 22, 7169-7192 (2020).
    [doi:10.1039/C9CP06869D](https://doi.org/10.1039/C9CP06869D)
- **Also found as:** `gfn2//gfnff`, `crest gfn2//gfnff`, `gfn2/gfnff`

<a id="crest-gfn1"></a>

### CREST level: GFN1-xTB

> The whole CREST search at GFN1-xTB: the other tight-binding level, at about GFN2's cost.

Metadynamics, optimizations and the ranking use [GFN1-xTB](#gfn1-xtb).

- **Good for:** A cross-check of a GFN2 ensemble.
- **Watch out:** The same run times as GFN2; GFN1's simpler electrostatics rank polar conformers differently.
- **Cost:** About that of CREST with GFN2-xTB.
- **Dispersion:** Yes: D3(BJ).
- **Solvation:** ALPB.
- **See also:** [CREST iMTD-GC](#crest), [GFN1-xTB](#gfn1-xtb)
- **References:**
  - Pracht, Bohle, Grimme. Automated exploration of the low-energy chemical space with fast quantum chemical
    methods. Phys. Chem. Chem. Phys. 22, 7169-7192 (2020).
    [doi:10.1039/C9CP06869D](https://doi.org/10.1039/C9CP06869D)
  - Grimme, Bannwarth, Shushkov. A robust and accurate tight-binding quantum chemical method for structures,
    vibrational frequencies, and noncovalent interactions of large molecular systems parametrized for all
    spd-block elements (Z = 1-86). J. Chem. Theory Comput. 13, 1989-2009 (2017).
    [doi:10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)
- **Also found as:** `crest gfn1`, `crest --gfn1`

## DFT functionals and HF

<a id="dft"></a>

### DFT and functional names

> Density functional theory gets the energy from the electron density; the functional's name says which
> approximation.

DFT computes the energy from the electron density. The unknown exchange-correlation part is approximated by a
functional. From cheap to expensive: GGA (density and its gradient: PBE, BLYP), meta-GGA (also the
kinetic-energy density: r2SCAN), hybrid (a fixed share of exact Hartree-Fock exchange: B3LYP 20 %, PBE0 25 %,
M06-2X 54 %) and range-separated hybrid (the share of exact exchange grows with the distance between
electrons: ωB97X-D). Suffixes name the [dispersion correction](#dispersion) (-D3BJ, -D4, -D, -V).

- **Good for:** Reading level names: 'ωB97X-D/6-31G\*' is functional/[basis set](#basis-sets);
  'B3LYP/6-311+G\*\*//MMFF' is a B3LYP single point at an MMFF geometry ([Single point vs
  optimization](#single-point)).
- **Watch out:** A level is functional + basis set + solvent model + geometry. The same functional with
  another basis, solvent model or geometry gives other numbers. GMTKN55, a broad benchmark of main-group
  chemistry, recommends dispersion corrections throughout and does not include B3LYP in its recommendations.
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N.
- **Dispersion:** Depends on the functional (see each entry).
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [ωB97X-D](#wb97x-d), [B3LYP-D3(BJ)](#b3lyp-d3bj), [Basis sets (what the names
  mean)](#basis-sets), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion)
- **References:**
  - Smith et al.. Psi4 1.4: open-source software for high-throughput quantum chemistry. J. Chem. Phys. 152,
    184108 (2020). [doi:10.1063/5.0006002](https://doi.org/10.1063/5.0006002)
  - Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme. A look at the density functional theory zoo with the
    advanced GMTKN55 database for general main group thermochemistry, kinetics and noncovalent interactions.
    Phys. Chem. Chem. Phys. 19, 32184-32215 (2017).
    [doi:10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G)
- **Also found as:** `density functional theory`, `functional`, `functionals`, `qm`, `psi4`

<a id="wb97x-d"></a>

### ωB97X-D

> Range-separated hybrid with its own dispersion term (Chai and Head-Gordon); QuantaBench's default
> functional.

A range-separated hybrid: about 22 % exact exchange between nearby electrons, rising to 100 % at long range
(switching length set by ω = 0.2 per bohr). The '-D' is its own damped atom-pair dispersion term, fitted
together with the functional; it is an older pairwise form, not Grimme's D3. Full exact exchange at long range
improves anions, charge transfer and orbital energies over global hybrids such as B3LYP.

- **Good for:** General organic chemistry: conformer energies, noncovalent contacts, dipoles and
  polarizabilities. Matching Spartan's ωB97X-D/6-31G\* results: at Spartan's geometry QuantaBench's Psi4 stage
  reproduces its energies to 4e-4 Eh and its dipoles to 0.001 D (docs/VALIDATION.md, VAL-2).
- **Watch out:** Its dispersion term predates D3(BJ) and D4. Psi4 computes its polarizability analytically,
  except in PCM.
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N.
- **Dispersion:** Yes: its own damped atom-pair term ('-D').
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [Spartan
  ωB97X-D/6-31G\*](#spartan-wb97x-d), [B3LYP-D3(BJ)](#b3lyp-d3bj), [DFT and functional names](#dft)
- **References:**
  - Chai, Head-Gordon. Long-range corrected hybrid density functionals with damped atom-atom dispersion
    corrections. Phys. Chem. Chem. Phys. 10, 6615 (2008).
    [doi:10.1039/B810189B](https://doi.org/10.1039/B810189B)
- **Also found as:** `wb97xd`, `ωb97xd`

<a id="b3lyp-d3bj"></a>

### B3LYP-D3(BJ)

> The classic global hybrid (20 % exact exchange) with Grimme's D3 dispersion and Becke-Johnson damping.

B3LYP mixes 20 % exact (Hartree-Fock) exchange with Becke's gradient-corrected exchange and the Lee-Yang-Parr
correlation functional, with three fitted mixing parameters. The -D3(BJ) suffix adds Grimme's D3 dispersion
with Becke-Johnson damping; without it B3LYP misses most of the London dispersion between nonbonded atoms
([B3LYP (no dispersion)](#b3lyp)).

- **Good for:** Comparing with the large literature that uses B3LYP, with the dispersion fixed; geometries and
  vibrational frequencies. Ranking the conformers of PFAS ions in PCM water: at def2-SVP it agreed with
  ωB97X-D (median ρ 0.95; its lowest within 4 kJ/mol of ωB97X-D's in 9 of 9 ions) at 0.72x the cost
  (docs/VALIDATION.md, RANK-1).
- **Watch out:** A global hybrid: less suited to anions and charge transfer than range-separated ones. GMTKN55
  does not include B3LYP in its recommendations. Needs `dftd3-python` in the Psi4 environment. Its agreement
  with ωB97X-D/def2-SVP on PFAS ions comes largely from the shared basis: against r2SCAN-3c in PCM, def2-SVP
  B3LYP-D3(BJ) favoured compact conformers (median ρ 0.60, 3 of 5 picks within 4 kJ/mol, against 0.94 and 5 of
  5 for g-xTB + GFN1-xTB CPCM; docs/VALIDATION.md, RANK-1).
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N. D3 itself adds almost nothing.
- **Dispersion:** Yes: D3 with Becke-Johnson damping.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [B3LYP (no dispersion)](#b3lyp), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion),
  [PBE0-D3(BJ)](#pbe0-d3bj), [DFT and functional names](#dft)
- **References:**
  - Becke. Density-functional thermochemistry. III. The role of exact exchange. J. Chem. Phys. 98, 5648-5652
    (1993). [doi:10.1063/1.464913](https://doi.org/10.1063/1.464913)
  - Stephens, Devlin, Chabalowski, Frisch. Ab initio calculation of vibrational absorption and circular
    dichroism spectra using density functional force fields. J. Phys. Chem. 98, 11623-11627 (1994).
    [doi:10.1021/j100096a001](https://doi.org/10.1021/j100096a001)
  - Grimme, Ehrlich, Goerigk. Effect of the damping function in dispersion corrected density functional
    theory. J. Comput. Chem. 32, 1456-1465 (2011). [doi:10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)
- **Also found as:** `b3lyp-d3(bj)`, `b3lyp-d3`, `b3lypd3bj`

<a id="b3lyp"></a>

### B3LYP (no dispersion)

> B3LYP without a dispersion correction, as in Spartan's B3LYP runs: underbinds folded and stacked structures.

Plain B3LYP is the functional of many older studies and of Spartan's B3LYP/6-311+G\*\* calculations. It has no
dispersion correction. London dispersion is the attraction between atoms that are close but not bonded, so a
folded conformer, where chain segments touch, loses an attraction an extended one does not need. Plain B3LYP
therefore tends to favour extended chains over folded ones. Even n-alkanes longer than about 17 carbons fold
into hairpins in the gas phase, held together by dispersion between the strands.

- **Good for:** Reproducing older or Spartan results. Type `b3lyp` in the Functional box; use b3lyp-d3bj for
  new work.
- **Watch out:** This matters for PFAS conformers. In QuantaBench's comparison, GFN2/ALPB (with D4) put
  compact, low-dipole ions lowest and Spartan's B3LYP/C-PCM single points put extended ones lowest, and the
  conformer choice is 88 % of the 4.35 D water-dipole gap (docs/VALIDATION.md, VAL-3). Dispersion is only a
  small part of that: at common GFN2 geometries D3(BJ) moves the energy difference between the two picks by
  about 10 kJ/mol, and ωB97X-D and B3LYP-D3(BJ) in PCM water side with Spartan's pick, not GFN2's (the DFT
  conformer test, VAL-3).
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N.
- **Dispersion:** No.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [B3LYP-D3(BJ)](#b3lyp-d3bj), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion),
  [Spartan B3LYP/6-311+G\*\* (C-PCM water)](#spartan-b3lyp)
- **References:**
  - Stephens, Devlin, Chabalowski, Frisch. Ab initio calculation of vibrational absorption and circular
    dichroism spectra using density functional force fields. J. Phys. Chem. 98, 11623-11627 (1994).
    [doi:10.1021/j100096a001](https://doi.org/10.1021/j100096a001)
  - Grimme, Hansen, Brandenburg, Bannwarth. Dispersion-corrected mean-field electronic structure methods.
    Chem. Rev. 116, 5105-5154 (2016).
    [doi:10.1021/acs.chemrev.5b00533](https://doi.org/10.1021/acs.chemrev.5b00533)
  - Lüttschwager, Wassermann, Mata, Suhm. The last globally stable extended alkane. Angew. Chem. Int. Ed. 52,
    463-466 (2013). [doi:10.1002/anie.201202894](https://doi.org/10.1002/anie.201202894)
- **Also found as:** `b3lyp plain`, `b3lyp without dispersion`

<a id="pbe0-d3bj"></a>

### PBE0-D3(BJ)

> Non-empirical global hybrid (PBE with 25 % exact exchange) plus D3(BJ) dispersion: a general-purpose choice.

PBE0 takes the PBE gradient-corrected functional, built from physical constraints rather than fitted to data,
and replaces 25 % of its exchange with exact exchange. D3(BJ) adds dispersion.

- **Good for:** Geometries, dipoles and polarizabilities; a second hybrid to check B3LYP or ωB97X-D results.
- **Watch out:** A global hybrid, with the same long-range limits as B3LYP for anions and charge transfer.
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N.
- **Dispersion:** Yes: D3 with Becke-Johnson damping.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [B3LYP-D3(BJ)](#b3lyp-d3bj), [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [DFT
  and functional names](#dft)
- **References:**
  - Perdew, Burke, Ernzerhof. Generalized gradient approximation made simple. Phys. Rev. Lett. 77, 3865-3868
    (1996). [doi:10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865)
  - Adamo, Barone. Toward reliable density functional methods without adjustable parameters: the PBE0 model.
    J. Chem. Phys. 110, 6158-6170 (1999). [doi:10.1063/1.478522](https://doi.org/10.1063/1.478522)
  - Grimme, Ehrlich, Goerigk. Effect of the damping function in dispersion corrected density functional
    theory. J. Comput. Chem. 32, 1456-1465 (2011). [doi:10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)
- **Also found as:** `pbe0-d3(bj)`, `pbe0`, `pbeh`, `pbe0-d3`

<a id="r2scan"></a>

### r2SCAN

> Meta-GGA without exact exchange: cheaper than hybrids; no long-range dispersion as listed (r2SCAN-D4 adds
> it).

A meta-GGA uses the kinetic-energy density as well as the density and its gradient, which lets it tell single
bonds, double bonds and weak contacts apart. r2SCAN (regularized-restored SCAN) keeps most of SCAN's exact
constraints and is numerically stable on ordinary integration grids. It has no exact exchange.

- **Good for:** Geometries and conformer energies of larger molecules when hybrids are too slow. The composite
  r2SCAN-3c was QuantaBench's check on the conformer reference for PFAS ions in PCM water. On 5 ions it agreed
  with ωB97X-D/def2-SVPD (median ρ 0.99) better than ωB97X-D/def2-SVP did (0.72). Of the cheap rankers, it
  agreed best with g-xTB + GFN1-xTB CPCM (ρ 0.94; docs/VALIDATION.md, RANK-1).
- **Watch out:** It captures some medium-range dispersion but not long-range dispersion: r2SCAN-D4 adds it
  (Psi4 needs `dftd4-python` for -d4 names). Without exact exchange, HOMO-LUMO gaps come out smaller than with
  hybrids. Psi4 uses a finite-field polarizability for meta-GGAs.
- **Cost:** No exact exchange: with density fitting, cheaper than a hybrid in the same basis. Not so the
  composite r2SCAN-3c (def2-mTZVPP basis, D4, gCP; Psi4 needs `mctc-gcp` on the PATH): in PCM it took 2x
  ωB97X-D/def2-SVP for a 37-atom PFAS ion in Psi4 1.9.1 (QuantaBench measured, RANK-1).
- **Dispersion:** Medium range only, from the functional itself.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [DFT and functional names](#dft)
- **References:**
  - Furness, Kaplan, Ning, Perdew, Sun. Accurate and numerically efficient r2SCAN meta-generalized gradient
    approximation. J. Phys. Chem. Lett. 11, 8208-8215 (2020).
    [doi:10.1021/acs.jpclett.0c02405](https://doi.org/10.1021/acs.jpclett.0c02405)
  - Ehlert, Huniar, Ning, Furness, Sun, Kaplan, Perdew, Brandenburg. r2SCAN-D4: dispersion corrected
    meta-generalized gradient approximation for general chemical applications. J. Chem. Phys. 154, 061101
    (2021). [doi:10.1063/5.0041008](https://doi.org/10.1063/5.0041008)
  - Grimme, Hansen, Ehlert, Mewes. r2SCAN-3c: a "Swiss army knife" composite electronic-structure method. J.
    Chem. Phys. 154, 064103 (2021). [doi:10.1063/5.0040021](https://doi.org/10.1063/5.0040021)
- **Also found as:** `r2-scan`, `r²scan`, `r2scan-3c`

<a id="m06-2x"></a>

### M06-2X

> Truhlar's highly parametrized hybrid meta-GGA (54 % exact exchange) for main-group thermochemistry,
> barriers.

M06-2X is one of the Minnesota 2006 functionals; '2X' means twice the exact exchange of M06 (54 %). It was
fitted to main-group thermochemistry, reaction barriers and noncovalent interactions, so its parameters
describe medium-range dispersion.

- **Good for:** Reaction energies and barriers of organic molecules; noncovalent complexes at short range.
- **Watch out:** No long-range dispersion: GMTKN55 found that Minnesota functionals still need a D3
  correction. Meta-GGAs of the M06 family are sensitive to the integration grid. Psi4 uses a finite-field
  polarizability.
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N. It needs a finer integration grid for stable results.
- **Dispersion:** Medium range only, through its parametrization.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion), [DFT and functional names](#dft)
- **References:**
  - Zhao, Truhlar. The M06 suite of density functionals for main group thermochemistry, thermochemical
    kinetics, noncovalent interactions, excited states, and transition elements: two new functionals and
    systematic testing of four M06-class functionals and 12 other functionals. Theor. Chem. Acc. 120, 215-241
    (2008). [doi:10.1007/s00214-007-0310-x](https://doi.org/10.1007/s00214-007-0310-x)
  - Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme. A look at the density functional theory zoo with the
    advanced GMTKN55 database for general main group thermochemistry, kinetics and noncovalent interactions.
    Phys. Chem. Chem. Phys. 19, 32184-32215 (2017).
    [doi:10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G)
  - Wheeler, Houk. Integration grid errors for meta-GGA-predicted reaction energies: origin of grid errors for
    the M06 suite of functionals. J. Chem. Theory Comput. 6, 395-404 (2010).
    [doi:10.1021/ct900639j](https://doi.org/10.1021/ct900639j)
- **Also found as:** `m062x`, `m06 2x`

<a id="hf"></a>

### Hartree-Fock (HF)

> Mean-field wavefunction theory without electron correlation: a baseline, not a level for production
> energies.

Each electron moves in the average field of the others. Exchange is exact, but electron correlation is missing
entirely, dispersion included. Correlated wavefunction methods start from HF, and the NDDO methods (PM6, PM7)
are parametrized HF.

- **Good for:** A baseline; Koopmans' estimate of the ionization energy (−ε\_HOMO); the HF/6-31G\*
  electrostatic potential that classic force-field charges are fitted to.
- **Watch out:** No correlation and no dispersion: conformer energies of flexible chains and noncovalent
  contacts are unreliable, and HOMO-LUMO gaps are much larger than DFT's. HF-3c (Psi4's composite: the MINIX
  basis, D3, gCP and a short-range basis correction; it needs `mctc-gcp` on the PATH) does not fix that for
  PFAS conformers: against ωB97X-D/def2-SVP in PCM its lowest conformer was within 4 kJ/mol in 10 of 17 ions,
  favouring extended ones, and against ωB97X-D/6-31G\* in 18 of 27 neutrals (docs/VALIDATION.md, RANK-1).
- **Cost:** Similar to a hybrid DFT calculation in the same basis. HF-3c: 0.2-0.3x ωB97X-D/def2-SVP for PFAS
  ions in PCM, where the PCM steps then dominate (QuantaBench measured, RANK-1).
- **Dispersion:** No.
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [DFT and functional names](#dft), [HOMO and LUMO energies](#homo-lumo)
- **References:**
  - Roothaan. New developments in molecular orbital theory. Rev. Mod. Phys. 23, 69-89 (1951).
    [doi:10.1103/RevModPhys.23.69](https://doi.org/10.1103/RevModPhys.23.69)
  - Koopmans. Über die Zuordnung von Wellenfunktionen und Eigenwerten zu den einzelnen Elektronen eines Atoms.
    Physica 1, 104-113 (1934).
    [doi:10.1016/S0031-8914(34)90011-2](https://doi.org/10.1016/S0031-8914%2834%2990011-2)
  - Sure, Grimme. Corrected small basis set Hartree-Fock method for large systems. J. Comput. Chem. 34,
    1672-1685 (2013). [doi:10.1002/jcc.23317](https://doi.org/10.1002/jcc.23317)
- **Also found as:** `hartree-fock`, `rhf`, `uhf`, `scf`, `hf-3c`

<a id="wb97m"></a>

### ωB97M-D3(BJ) / ωB97M-V

> Head-Gordon's range-separated hybrid meta-GGA; AIMNet2 was trained on ωB97M with D3(BJ) dispersion.

ωB97M-V combines range-separated exact exchange (as in ωB97X-D) with a meta-GGA form and the VV10 nonlocal
correlation functional for dispersion; its terms were chosen by a combinatorial search over many forms. The
D3(BJ) variant uses D3(BJ) instead of VV10. AIMNet2's training data are ωB97M/def2-TZVPP with D3(BJ), so
AIMNet2's numbers aim at this level, not at B3LYP or ωB97X-D.

- **Good for:** Knowing what [AIMNet2](#aimnet2) imitates. Psi4 runs ωB97M-V when you type `wb97m-v` in the
  Functional box.
- **Watch out:** VV10 makes Psi4 use a finite-field polarizability and adds grid work.
- **Cost:** Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23
  atoms, 313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads,
  docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number of basis
  functions N.
- **Dispersion:** Yes: VV10 (ωB97M-V) or D3(BJ).
- **Solvation:** PCM in Psi4 (single points).
- **See also:** [AIMNet2](#aimnet2), [ωB97X-D](#wb97x-d), [Dispersion corrections (D3, D3(BJ), D4,
  -D)](#dispersion)
- **References:**
  - Mardirossian, Head-Gordon. ωB97M-V: a combinatorially optimized, range-separated hybrid, meta-GGA density
    functional with VV10 nonlocal correlation. J. Chem. Phys. 144, 214110 (2016).
    [doi:10.1063/1.4952647](https://doi.org/10.1063/1.4952647)
  - Anstine, Zubatyuk, Isayev. AIMNet2: a neural network potential to meet your neutral, charged, organic, and
    elemental-organic needs. Chem. Sci. 16, 10228-10244 (2025).
    [doi:10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
- **Also found as:** `wb97m-v`, `wb97m-d3bj`, `wb97m-d3(bj)`, `wb97m-d3`, `ωb97m-v`

## Basis sets

<a id="basis-sets"></a>

### Basis sets (what the names mean)

> The functions molecular orbitals are built from: more and wider functions give better answers at a steep
> cost.

Each molecular orbital is a sum of atom-centred Gaussian functions. Split-valence bases ('SV', 6-31G) give
each valence orbital two functions of different size, triple-zeta bases ('TZV', 6-311G) three. Polarization
functions ('P', '\*', '(d,p)') add higher angular momentum (d on C, N, O, F; p on H) so orbitals can bend and
bonds can polarize. Diffuse functions ('D', '+', 'aug-') are wide, slowly decaying functions for electrons far
from the nuclei.

- **Good for:** def2-SVP for screening, def2-TZVP for final energies, a diffuse basis (def2-SVPD, 6-31+G\*,
  aug-cc-pVDZ) for anions, polarizabilities and ESP surfaces.
- **Watch out:** Anions need diffuse functions: the extra electron occupies a spread-out orbital that a
  compact basis squeezes, which raises the anion's energy and shrinks its polarizability (carboxylates,
  sulfonates). Psi4 and Spartan use 6 Cartesian d functions in 6-31G\* (QuantaBench checked the basis-function
  counts, docs/VALIDATION.md); def2 and Dunning sets use 5 spherical ones.
- **Cost:** Basis functions for PFOA (C8HF15O2, 26 atoms): def2-SVP 355, 6-31G\* 377, 6-31+G\* 477,
  6-311+G\*\* 556, def2-SVPD 559, aug-cc-pVDZ 584, def2-TZVP 781 (6 Cartesian d functions in 6-31G\* and
  6-31+G\*, 5 spherical ones in the others). QuantaBench's Psi4 time grows about as N^2.5
  (docs/VALIDATION.md), so def2-TZVP costs about 7x def2-SVP.
- **Dispersion:** Not applicable (a basis set does not add dispersion).
- **Solvation:** Not applicable.
- **See also:** [def2-SVP](#def2-svp), [def2-TZVP](#def2-tzvp), [def2-SVPD](#def2-svpd), [6-31G\*](#6-31gs),
  [aug-cc-pVDZ](#aug-cc-pvdz)
- **References:**
  - Weigend, Ahlrichs. Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence
    quality for H to Rn: design and assessment of accuracy. Phys. Chem. Chem. Phys. 7, 3297 (2005).
    [doi:10.1039/B508541A](https://doi.org/10.1039/B508541A)
  - Hehre, Ditchfield, Pople. Self-consistent molecular orbital methods. XII. Further extensions of
    Gaussian-type basis sets for use in molecular orbital studies of organic molecules. J. Chem. Phys. 56,
    2257-2261 (1972). [doi:10.1063/1.1677527](https://doi.org/10.1063/1.1677527)
  - Clark, Chandrasekhar, Spitznagel, Schleyer. Efficient diffuse function-augmented basis sets for anion
    calculations. III. The 3-21+G basis set for first-row elements, Li-F. J. Comput. Chem. 4, 294-301 (1983).
    [doi:10.1002/jcc.540040303](https://doi.org/10.1002/jcc.540040303)
- **Also found as:** `basis`, `basis set`, `diffuse functions`, `polarization functions`, `split valence`

<a id="def2-svp"></a>

### def2-SVP

> Ahlrichs split-valence double-zeta with polarization: the standard fast DFT basis (355 functions for PFOA).

Two functions per valence orbital, one d set on heavy atoms and one p set on hydrogen. The def2 family
(Weigend and Ahlrichs) is balanced across the periodic table, H to Rn, with effective core potentials beyond
Kr.

- **Good for:** Geometries and screening at hybrid-DFT level; QuantaBench's default QM basis.
- **Watch out:** No diffuse functions, so anions come out too compact. QuantaBench measured, against
  ωB97X-D/6-31G\* at the same geometry for TFA and PFPrA: dipoles −0.05 to −0.07 D, HOMO/LUMO −0.21 to −0.25
  eV, polarizability −4 % (docs/VALIDATION.md, VAL-2). For conformers of PFAS ions in PCM water,
  ωB97X-D/def2-SVP favoured compact conformers over extended ones by 11-13 kJ/mol on average relative to
  def2-SVPD and r2SCAN-3c (docs/VALIDATION.md, RANK-1): use a diffuse or triple-zeta basis to rank them.
- **Cost:** PFHpA (313 functions) at ωB97X-D: SCF 101 s, full protocol 507 s on 2 threads (QuantaBench
  measured).
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [def2-SVPD](#def2-svpd),
  [def2-TZVP](#def2-tzvp)
- **References:**
  - Weigend, Ahlrichs. Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence
    quality for H to Rn: design and assessment of accuracy. Phys. Chem. Chem. Phys. 7, 3297 (2005).
    [doi:10.1039/B508541A](https://doi.org/10.1039/B508541A)
- **Also found as:** `def2svp`, `svp`

<a id="def2-svpd"></a>

### def2-SVPD

> def2-SVP plus diffuse functions tuned for response properties: for anions and polarizabilities.

def2-SVP with diffuse functions (Rappoport and Furche) chosen to give accurate polarizabilities and other
response properties at small cost.

- **Good for:** Anions (carboxylates, sulfonates), polarizabilities and ESP surfaces at DFT level. Conformer
  energies of PFAS ions: on the ion where ωB97X-D/def2-SVP and r2SCAN-3c disagreed most, def2-SVPD moved
  ωB97X-D to within 8 kJ/mol MAE of r2SCAN-3c, from 16 (docs/VALIDATION.md, RANK-1).
- **Watch out:** Diffuse functions slow the SCF and can make it harder to converge.
- **Cost:** 559 functions for PFOA: about 3x the def2-SVP time. ωB97X-D/def2-SVPD in PCM took 740-850 s for a
  37-atom PFAS ion on 4 threads, about 3x def2-SVP (QuantaBench measured, RANK-1).
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [def2-SVP](#def2-svp)
- **References:**
  - Rappoport, Furche. Property-optimized Gaussian basis sets for molecular response calculations. J. Chem.
    Phys. 133, 134105 (2010). [doi:10.1063/1.3484283](https://doi.org/10.1063/1.3484283)
- **Also found as:** `def2svpd`, `svpd`

<a id="def2-tzvp"></a>

### def2-TZVP

> Ahlrichs triple-zeta with polarization: close to converged DFT energies; about 7x the cost of def2-SVP.

Three functions per valence orbital, two d sets and one f set on heavy atoms (C to F) and one p set on
hydrogen.

- **Good for:** Final energies and properties of neutral molecules at DFT level.
- **Watch out:** No diffuse functions, so anions and polarizabilities still need a diffuse basis.
- **Cost:** 781 functions for PFOA: about 7x the def2-SVP time.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [def2-SVP](#def2-svp)
- **References:**
  - Weigend, Ahlrichs. Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence
    quality for H to Rn: design and assessment of accuracy. Phys. Chem. Chem. Phys. 7, 3297 (2005).
    [doi:10.1039/B508541A](https://doi.org/10.1039/B508541A)
- **Also found as:** `def2tzvp`, `tzvp`

<a id="6-31gs"></a>

### 6-31G\*

> Pople split-valence with d functions on heavy atoms: the classic basis, and Spartan's ωB97X-D/6-31G\* level.

Core orbitals use one function (a contraction of 6 Gaussians), valence orbitals two (3 + 1 Gaussians); '\*'
adds one set of d functions on non-hydrogen atoms. Also written 6-31G(d).

- **Good for:** Like-for-like comparisons with Spartan's ωB97X-D/6-31G\* results; quick DFT geometries.
- **Watch out:** No polarization on hydrogen (6-31G\*\* adds it) and no diffuse functions for anions.
- **Cost:** 377 functions for PFOA. PFHpA (332) at ωB97X-D: SCF 127 s, full protocol 533 s on 2 threads
  (QuantaBench measured).
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [6-31+G\*](#6-31pgs), [Spartan
  ωB97X-D/6-31G\*](#spartan-wb97x-d)
- **References:**
  - Hehre, Ditchfield, Pople. Self-consistent molecular orbital methods. XII. Further extensions of
    Gaussian-type basis sets for use in molecular orbital studies of organic molecules. J. Chem. Phys. 56,
    2257-2261 (1972). [doi:10.1063/1.1677527](https://doi.org/10.1063/1.1677527)
  - Hariharan, Pople. The influence of polarization functions on molecular orbital hydrogenation energies.
    Theor. Chim. Acta 28, 213-222 (1973). [doi:10.1007/BF00533485](https://doi.org/10.1007/BF00533485)
- **Also found as:** `6-31g(d)`, `631g*`, `6-31gs`

<a id="6-31pgs"></a>

### 6-31+G\*

> 6-31G\* plus diffuse s and p functions on heavy atoms: the cheapest Pople basis for anions.

'+' adds one diffuse s and one diffuse p function on every non-hydrogen atom. Also written 6-31+G(d).

- **Good for:** Anions at modest cost.
- **Watch out:** Still no functions beyond s on hydrogen.
- **Cost:** 477 functions for PFOA: about 2x the def2-SVP time.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [6-31G\*](#6-31gs)
- **References:**
  - Hehre, Ditchfield, Pople. Self-consistent molecular orbital methods. XII. Further extensions of
    Gaussian-type basis sets for use in molecular orbital studies of organic molecules. J. Chem. Phys. 56,
    2257-2261 (1972). [doi:10.1063/1.1677527](https://doi.org/10.1063/1.1677527)
  - Clark, Chandrasekhar, Spitznagel, Schleyer. Efficient diffuse function-augmented basis sets for anion
    calculations. III. The 3-21+G basis set for first-row elements, Li-F. J. Comput. Chem. 4, 294-301 (1983).
    [doi:10.1002/jcc.540040303](https://doi.org/10.1002/jcc.540040303)
- **Also found as:** `6-31+g(d)`, `631+g*`

<a id="6-311pgss"></a>

### 6-311+G\*\*

> Pople triple-split valence with diffuse and polarization functions: the basis of Spartan's B3LYP runs.

Valence orbitals get three functions; '+' adds diffuse s and p on heavy atoms and '\*\*' adds d on heavy atoms
and p on hydrogen. Also written 6-311+G(d,p).

- **Good for:** Like-for-like comparisons with Spartan's B3LYP/6-311+G\*\* water results; anions.
- **Watch out:** No f functions, and it costs about as much as def2-SVPD.
- **Cost:** 556 functions for PFOA: about 3x the def2-SVP time.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [Spartan B3LYP/6-311+G\*\* (C-PCM
  water)](#spartan-b3lyp)
- **References:**
  - Krishnan, Binkley, Seeger, Pople. Self-consistent molecular orbital methods. XX. A basis set for
    correlated wave functions. J. Chem. Phys. 72, 650-654 (1980).
    [doi:10.1063/1.438955](https://doi.org/10.1063/1.438955)
  - Clark, Chandrasekhar, Spitznagel, Schleyer. Efficient diffuse function-augmented basis sets for anion
    calculations. III. The 3-21+G basis set for first-row elements, Li-F. J. Comput. Chem. 4, 294-301 (1983).
    [doi:10.1002/jcc.540040303](https://doi.org/10.1002/jcc.540040303)
- **Also found as:** `6-311+g(d,p)`, `6311+g**`

<a id="aug-cc-pvdz"></a>

### aug-cc-pVDZ

> Dunning correlation-consistent double-zeta with diffuse functions on every atom, hydrogen included.

cc-pVnZ sets were designed so that results converge systematically from DZ to TZ to QZ in correlated
calculations. 'aug-' adds one diffuse function of each angular momentum on every atom.

- **Good for:** Anions, polarizabilities and weak complexes; the first step of a basis-set convergence series.
- **Watch out:** Diffuse functions on all atoms: slower, and in large molecules nearly linearly dependent
  functions can stall the SCF. For DFT, def2-SVPD costs about the same and was tuned for response properties.
- **Cost:** 584 functions for PFOA: about 3.5x the def2-SVP time.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [Basis sets (what the names mean)](#basis-sets), [def2-SVPD](#def2-svpd)
- **References:**
  - Dunning. Gaussian basis sets for use in correlated molecular calculations. I. The atoms boron through neon
    and hydrogen. J. Chem. Phys. 90, 1007-1023 (1989).
    [doi:10.1063/1.456153](https://doi.org/10.1063/1.456153)
  - Kendall, Dunning, Harrison. Electron affinities of the first-row atoms revisited. Systematic basis sets
    and wave functions. J. Chem. Phys. 96, 6796-6806 (1992).
    [doi:10.1063/1.462569](https://doi.org/10.1063/1.462569)
- **Also found as:** `augccpvdz`, `aug-cc-pvdz (dunning)`

## Implicit solvation

<a id="solvation"></a>

### Implicit solvation

> The solvent as a polarizable continuum around the molecule; one name ('water') maps onto each engine's
> model.

An implicit (continuum) model replaces the solvent molecules with a dielectric medium outside a cavity around
the solute. The solute's charges polarize the medium, which acts back on them (the reaction field); some
models add empirical terms for cavity formation and dispersion with the solvent. QuantaBench takes one solvent
name and maps it onto each engine's model: [ALPB](#alpb) in tblite, xtb and CREST, [COSMO](#cosmo) (from ε) in
MOPAC, [PCM](#pcm) in Psi4. Engines without one (dxtb, AIMNet2, MMFF94s) run in the gas phase and the notes
say so.

- **Good for:** Ions and polar molecules, where water changes conformer rankings and raises dipoles: at the
  same geometry PCM water raised the ωB97X-D/6-31G\* dipoles of TFA and PFPrA by 26-28 % (QuantaBench
  measured, docs/VALIDATION.md).
- **Watch out:** No explicit hydrogen bonds to water and no counter-ions. The RDKit conformer search is always
  in the gas phase; only CREST searches in ALPB. Solvation energies from different models are not
  interchangeable.
- **Cost:** ALPB adds little to an xTB calculation. In Psi4, PCM rules out analytic gradients and
  polarizabilities (gas-phase optimizations, finite fields).
- **Dispersion:** Not applicable.
- **Solvation:** The subject of this family.
- **See also:** [ALPB](#alpb), [GBSA](#gbsa), [COSMO](#cosmo), [PCM (IEF-PCM, C-PCM)](#pcm), [SMD](#smd), [Gas
  phase](#gas-phase)
- **References:**
  - Cramer, Truhlar. Implicit solvation models: equilibria, structure, spectra, and dynamics. Chem. Rev. 99,
    2161-2200 (1999). [doi:10.1021/cr960149m](https://doi.org/10.1021/cr960149m)
  - Tomasi, Mennucci, Cammi. Quantum mechanical continuum solvation models. Chem. Rev. 105, 2999-3094 (2005).
    [doi:10.1021/cr9904009](https://doi.org/10.1021/cr9904009)
- **Also found as:** `solvent`, `implicit solvent`, `continuum solvation`, `water`

<a id="gas-phase"></a>

### Gas phase

> The isolated molecule in vacuum: no solvent, and nothing screens an ion's charge.

With no solvent, or with an engine that has no solvent model, every step treats the molecule as isolated. This
is the reference state of most benchmark data and of Spartan's PM6 and ωB97X-D/6-31G\* results.

- **Good for:** Neutral molecules; comparison with gas-phase reference data.
- **Watch out:** Ions are unscreened: a multiply charged anion can be unbound in the gas phase (its HOMO above
  zero), and its xTB polarizability can then diverge. QuantaBench reports ions' dipoles about the centre of
  mass, because an ion's dipole depends on the origin.
- **Cost:** None.
- **Dispersion:** Not applicable.
- **Solvation:** None.
- **See also:** [Implicit solvation](#solvation)
- **Also found as:** `gas`, `vacuum`, `none`

<a id="alpb"></a>

### ALPB

> Analytical linearized Poisson-Boltzmann: the xTB solvent model (tblite, xtb, CREST), fitted per solvent.

ALPB is a generalized-Born-type model: it estimates the continuum's reaction field from atomic Born radii,
with an analytical correction for the solute's shape, and adds a surface-area term for cavity and dispersion
effects. It is parametrized for each method (GFN1, GFN2, GFN-FF) and for more than 20 solvents. Its authors
report hydration free energies with GFN1-xTB(ALPB) within 1.4 kcal/mol of experiment on average, and
octanol-water log K within about 0.65 with GFN2-xTB(ALPB).

- **Good for:** Fast solvated optimizations and conformer ranking at xTB level; QuantaBench's water runs.
- **Watch out:** A continuum: no explicit hydrogen bonds to water. GFN2/ALPB and B3LYP/C-PCM rank folded and
  extended PFAS ions differently (VAL-3). Between conformers of one PFAS ion, ALPB's solvation energy
  differences missed IEF-PCM's (at ωB97X-D/def2-SVP) by 20 kJ/mol on average, favouring the compact conformer;
  tblite's CPCM with GFN1-xTB missed them by 4 (docs/VALIDATION.md, RANK-1). ALPB's non-polar terms are not
  the cause: its electrostatic term alone does no better.
- **Cost:** Little more than the gas-phase xTB calculation.
- **Dispersion:** Not applicable.
- **Solvation:** Yes: more than 20 solvents.
- **See also:** [GBSA](#gbsa), [Implicit solvation](#solvation), [GFN2-xTB](#gfn2-xtb)
- **References:**
  - Ehlert, Stahn, Spicher, Grimme. Robust and efficient implicit solvation model for fast semiempirical
    methods. J. Chem. Theory Comput. 17, 4250-4261 (2021).
    [doi:10.1021/acs.jctc.1c00471](https://doi.org/10.1021/acs.jctc.1c00471)
- **Also found as:** `alpb solvation`, `--alpb`

<a id="gbsa"></a>

### GBSA

> Generalized Born + solvent-accessible surface area: the older xTB solvent model, which ALPB replaced.

Generalized Born estimates the electrostatic solvation energy from pairwise atomic terms with effective Born
radii; the SASA term adds cavity and dispersion effects in proportion to the exposed surface. The CREST
documentation calls ALPB an improved version of GBSA.

- **Good for:** Reproducing older xtb results.
- **Watch out:** tblite has no GBSA parameters for some solvents (ethanol, ...) and falls back to ALPB with a
  note.
- **Cost:** Like ALPB.
- **Dispersion:** Not applicable.
- **Solvation:** Yes (fewer solvents than ALPB).
- **See also:** [ALPB](#alpb), [Implicit solvation](#solvation)
- **References:**
  - Still, Tempczyk, Hawley, Hendrickson. Semianalytical treatment of solvation for molecular mechanics and
    dynamics. J. Am. Chem. Soc. 112, 6127-6129 (1990).
    [doi:10.1021/ja00172a038](https://doi.org/10.1021/ja00172a038)
  - Ehlert, Stahn, Spicher, Grimme. Robust and efficient implicit solvation model for fast semiempirical
    methods. J. Chem. Theory Comput. 17, 4250-4261 (2021).
    [doi:10.1021/acs.jctc.1c00471](https://doi.org/10.1021/acs.jctc.1c00471)
- **Also found as:** `gb`, `generalized born`, `--gbsa`

<a id="cosmo"></a>

### COSMO

> Conductor-like screening model (MOPAC): surface charges for a perfect conductor, scaled to the solvent's ε.

COSMO first finds the surface charges that would screen the solute completely in a conductor, then scales them
down for a finite dielectric constant. C-PCM uses the same idea. QuantaBench passes MOPAC only the dielectric
constant (`EPS=`), so no solvent-specific parameters are involved.

- **Good for:** Solvated PM7 and PM6 energies; the COSMO area and volume descriptors (MOPAC computes them in
  the gas phase too).
- **Watch out:** MOPAC's COSMO is not smooth: energies and gradients jump as surface segments switch, so
  optimizations in COSMO can stop short and COSMO frequencies are approximate (CF3COO⁻ in water: a −28 cm⁻¹
  spurious mode, docs/ENGINES.md).
- **Cost:** Small compared with the NDDO calculation.
- **Dispersion:** Not applicable.
- **Solvation:** Yes: any solvent, from its dielectric constant.
- **See also:** [PCM (IEF-PCM, C-PCM)](#pcm), [PM7](#pm7), [Implicit solvation](#solvation)
- **References:**
  - Klamt, Schüürmann. COSMO: a new approach to dielectric screening in solvents with explicit expressions for
    the screening energy and its gradient. J. Chem. Soc., Perkin Trans. 2 799-805 (1993).
    [doi:10.1039/P29930000799](https://doi.org/10.1039/P29930000799)
- **Also found as:** `conductor-like screening model`

<a id="pcm"></a>

### PCM (IEF-PCM, C-PCM)

> Polarizable continuum model: Psi4's IEF-PCM, Spartan's C-PCM and tblite's CPCM; the DFT-level continuum.

PCM puts apparent charges on the surface of a cavity made of atomic spheres so that they reproduce the
dielectric's reaction field. IEF-PCM (Psi4, through PCMSolver) solves the dielectric problem; C-PCM (Spartan)
and COSMO treat the solvent as a conductor and scale the result, which changes little for water (ε about 78).
tblite's CPCM solves it with the ddX library, from the dielectric constant alone.

- **Good for:** DFT single points in water: dipoles, orbital energies, ESP and polarizabilities in the
  reaction field. tblite's CPCM is a cheap stand-in for the solvation part of a DFT/PCM conformer ranking:
  with GFN1-xTB charges its solvation energy differences between conformers of PFAS ions followed IEF-PCM's to
  3.9 kJ/mol MAE, with GFN2-xTB to 9.7 (docs/VALIDATION.md, RANK-1).
- **Watch out:** Electrostatics only in QuantaBench's setups (no cavity or dispersion terms, unlike
  [SMD](#smd)), so PCM energies are not solvation free energies. Psi4 has no analytic PCM gradients: QM
  optimizations and frequencies run in the gas phase, and polarizabilities use finite fields. PCMSolver has no
  DMF, n-hexane, diethyl ether, octanol or ethyl acetate (gas phase with a note).
- **Cost:** In QuantaBench's Psi4 runs, PCM water took about 1.7x the gas-phase time for PFHpA at
  ωB97X-D/def2-SVP (857 vs 507 s), the extra time being the finite-field polarizability (docs/VALIDATION.md).
- **Dispersion:** Not applicable.
- **Solvation:** Yes: the solvents PCMSolver knows (Psi4); any ε (tblite CPCM).
- **See also:** [COSMO](#cosmo), [SMD](#smd), [Implicit solvation](#solvation)
- **References:**
  - Tomasi, Mennucci, Cammi. Quantum mechanical continuum solvation models. Chem. Rev. 105, 2999-3094 (2005).
    [doi:10.1021/cr9904009](https://doi.org/10.1021/cr9904009)
  - Barone, Cossi. Quantum calculation of molecular energies and energy gradients in solution by a conductor
    solvent model. J. Phys. Chem. A 102, 1995-2001 (1998).
    [doi:10.1021/jp9716997](https://doi.org/10.1021/jp9716997)
  - Nottoli, Herbst, Mikhalev, Jha, Lipparini, Stamm. ddX: polarizable continuum solvation from small
    molecules to proteins. WIREs Comput. Mol. Sci. 14, e1726 (2024).
    [doi:10.1002/wcms.1726](https://doi.org/10.1002/wcms.1726)
- **Also found as:** `cpcm`, `c-pcm`, `iefpcm`, `ief-pcm`, `ddcosmo`, `pcm water`

<a id="smd"></a>

### SMD

> Truhlar's solvation model: PCM-type electrostatics plus fitted surface terms, for solvation free energies.

SMD adds to an IEF-PCM-type electrostatic term a CDS term (cavitation, dispersion, solvent structure) from
atomic surface tensions fitted to experimental solvation free energies in many solvents. Its authors report
mean unsigned errors of 0.6-1.0 kcal/mol for neutral solutes and about 4 kcal/mol for ions.

- **Good for:** Solvation free energies and partition coefficients at DFT level, in programs that have it.
- **Watch out:** QuantaBench's Psi4 stage runs IEF-PCM, not SMD: keep that in mind when comparing with SMD
  numbers from other programs.
- **Cost:** Like PCM.
- **Dispersion:** Not applicable.
- **Solvation:** Yes (not in QuantaBench).
- **See also:** [PCM (IEF-PCM, C-PCM)](#pcm), [Implicit solvation](#solvation)
- **References:**
  - Marenich, Cramer, Truhlar. Universal solvation model based on solute electron density and on a continuum
    model of the solvent defined by the bulk dielectric constant and atomic surface tensions. J. Phys. Chem. B
    113, 6378-6396 (2009). [doi:10.1021/jp810292n](https://doi.org/10.1021/jp810292n)
- **Also found as:** `smd solvation`

## Optimizers

<a id="lbfgs"></a>

### L-BFGS (ASE)

> Quasi-Newton optimizer that estimates the curvature from recent steps; the default, reliable for molecules.

Limited-memory BFGS keeps the last steps and gradient changes (100 in ASE) to estimate the curvature and takes
Newton-like steps, at most 0.2 Å per atom. It stops when the largest atomic force is below Force convergence
(0.03 eV/Å by default).

- **Good for:** Every engine; the batched GPU refinement has a batched version of it.
- **Watch out:** Each step is one engine call: with external programs (xtb, MOPAC) their own optimizer is
  several times faster.
- **Cost:** One energy and gradient per step; tens of steps for a flexible molecule.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [FIRE (ASE)](#fire), [Engine's own optimizer](#native-optimizer)
- **References:**
  - Liu, Nocedal. On the limited memory BFGS method for large scale optimization. Math. Program. 45, 503-528
    (1989). [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
- **Also found as:** `l-bfgs`, `bfgs`

<a id="fire"></a>

### FIRE (ASE)

> Fast inertial relaxation engine: damped dynamics toward the minimum; tolerant of rough surfaces, often
> slower.

FIRE moves the atoms like a damped molecular-dynamics run, speeding up while the force points downhill and
stopping when it turns uphill. It needs no curvature estimate.

- **Good for:** Rough or noisy energy surfaces (COSMO, strained starting geometries) where L-BFGS steps go
  wrong.
- **Watch out:** Often takes more steps than L-BFGS on well-behaved molecules.
- **Cost:** One energy and gradient per step.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [L-BFGS (ASE)](#lbfgs)
- **References:**
  - Bitzek, Koskinen, Gähler, Moseler, Gumbsch. Structural relaxation made simple. Phys. Rev. Lett. 97, 170201
    (2006). [doi:10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201)
- **Also found as:** `fire optimizer`

<a id="native-optimizer"></a>

### Engine's own optimizer

> The program's own optimizer in one run (xtb, MOPAC): several times faster than ASE stepping it; else L-BFGS.

xtb optimizes in approximate normal coordinates (ANCopt) and MOPAC with its own optimizer, each in one program
run instead of one run per step. QuantaBench converts Force convergence into the program's gradient-norm
criterion so that the result is at least as tight. Engines without one (dxtb, tblite, AIMNet2, MMFF94s) use
L-BFGS.

- **Good for:** Refinements with `xtb-gfn2`, `xtb-gfn1`, `gfnff`, `mopac-pm7` and `mopac-pm6`.
- **Watch out:** xtb exits successfully even when it runs out of cycles; QuantaBench reads its log and checks
  the final forces.
- **Cost:** xtb: 4-10x faster than ASE stepping xtb; MOPAC: 3-15x (docs/ENGINES.md).
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [L-BFGS (ASE)](#lbfgs)
- **Also found as:** `native`, `ancopt`, `engine optimizer`

## Spartan's levels

<a id="spartan-b3lyp"></a>

### Spartan B3LYP/6-311+G\*\* (C-PCM water)

> Spartan's water numbers: B3LYP without dispersion, 6-311+G\*\*, C-PCM water, single points at MMFF
> geometries.

In the Spartan '24 results QuantaBench is validated against (docs/VALIDATION.md), the charged molecules have
B3LYP/6-311+G\*\* single points in C-PCM water (dielectric 78.3) at MMFF geometries. They give the water
dipoles, HOMO/LUMO and ESP values that QuantaBench is compared with. The functional has no dispersion
correction ([B3LYP (no dispersion)](#b3lyp)).

- **Good for:** Knowing what you compare with. QuantaBench's closest setting is the QM stage with
  `b3lyp/6-311+g**` in water (Psi4 uses IEF-PCM, not C-PCM).
- **Watch out:** QuantaBench does not try to match these numbers exactly: its levels include dispersion.
  GFN2/ALPB reproduces these dipoles to 0.52 D MAE at Spartan's geometries (61 ions), but QuantaBench's own
  lowest conformers differ by 4.35 D MAE, 88 % of it from the conformer choice (VAL-3).
- **Cost:** Not applicable (Spartan's results).
- **Dispersion:** No.
- **Solvation:** C-PCM water (ε 78.3).
- **See also:** [B3LYP (no dispersion)](#b3lyp), [PCM (IEF-PCM, C-PCM)](#pcm), [6-311+G\*\*](#6-311pgss),
  [Dispersion corrections (D3, D3(BJ), D4, -D)](#dispersion)
- **References:**
  - Stephens, Devlin, Chabalowski, Frisch. Ab initio calculation of vibrational absorption and circular
    dichroism spectra using density functional force fields. J. Phys. Chem. 98, 11623-11627 (1994).
    [doi:10.1021/j100096a001](https://doi.org/10.1021/j100096a001)
  - Barone, Cossi. Quantum calculation of molecular energies and energy gradients in solution by a conductor
    solvent model. J. Phys. Chem. A 102, 1995-2001 (1998).
    [doi:10.1021/jp9716997](https://doi.org/10.1021/jp9716997)
- **Also found as:** `spartan b3lyp`, `spartan water`, `spartan dft water`

<a id="spartan-wb97x-d"></a>

### Spartan ωB97X-D/6-31G\*

> Spartan's gas-phase DFT level for the Essential 76 PFAS; QuantaBench's Psi4 stage reproduces it at the same
> geometry.

Spartan's ωB97X-D/6-31G\* gas-phase single points on its conformers, with energies and dipoles per conformer.
Its geometries come from DLFFG1, a neural-network estimate of ωB97X-D/6-31G\* geometries, and its ranking from
DLXD2XV, a neural-network estimate of ωB97X-V/6-311+G(2df,2p). At Spartan's geometry QuantaBench's Psi4 stage
gives the same energies to 4e-4 Eh and dipoles to 0.001 D (VAL-2).

- **Good for:** Like-for-like checks: the QM stage with `wb97x-d/6-31g*` in the gas phase.
- **Watch out:** For TFA and PFPrA Spartan's documents hold one anti acid conformer each, 17-19 kJ/mol above
  the syn one at GFN2 (VAL-3).
- **Cost:** Not applicable (Spartan's results).
- **Dispersion:** Yes: ωB97X-D's own term.
- **Solvation:** Gas phase.
- **See also:** [ωB97X-D](#wb97x-d), [6-31G\*](#6-31gs)
- **References:**
  - Chai, Head-Gordon. Long-range corrected hybrid density functionals with damped atom-atom dispersion
    corrections. Phys. Chem. Chem. Phys. 10, 6615 (2008).
    [doi:10.1039/B810189B](https://doi.org/10.1039/B810189B)
- **Also found as:** `spartan wb97x-d`, `dlxd2xv`, `spartan est. dft`

<a id="spartan-pm6"></a>

### Spartan PM6

> Spartan's gas-phase semi-empirical level: plain PM6, without dispersion or hydrogen-bond corrections.

Most neutral molecules have PM6 gas-phase optimizations with Spartan's QSAR properties (dipole, HOMO/LUMO,
areas, volumes). Spartan's keyword is plain PM6.

- **Good for:** Knowing what you compare with in the gas phase.
- **Watch out:** QuantaBench's `mopac-pm6` is [PM6-D3H4X](#pm6-d3h4x), so its geometries and energies differ
  from Spartan's PM6 where dispersion or hydrogen bonds matter. QuantaBench measured GFN2's LUMO about 5.5 eV
  below PM6's and its HOMO 1.1 eV below (gap 4.4 eV smaller, r 0.80-0.88), and gas dipoles 1.41 D MAE from
  PM6's (r 0.65) (docs/dev/BACKLOG.md, VAL-3).
- **Cost:** Not applicable (Spartan's results).
- **Dispersion:** No.
- **Solvation:** Gas phase.
- **See also:** [PM6-D3H4X](#pm6-d3h4x), [HOMO and LUMO energies](#homo-lumo)
- **References:**
  - Stewart. Optimization of parameters for semiempirical methods V: modification of NDDO approximations and
    application to 70 elements. J. Mol. Model. 13, 1173-1213 (2007).
    [doi:10.1007/s00894-007-0233-4](https://doi.org/10.1007/s00894-007-0233-4)
- **Also found as:** `spartan pm6`, `pm6 (spartan)`

## Concepts

<a id="dispersion"></a>

### Dispersion corrections (D3, D3(BJ), D4, -D)

> Adds the London attraction most density functionals miss: it matters for folded chains, stacking,
> conformers.

London dispersion is the attraction between atoms that are close but not bonded, from correlated fluctuations
of their electrons; it falls off as 1/R⁶. Common functionals (B3LYP, PBE0) describe it poorly, so a correction
adds damped −C6/R⁶ and −C8/R⁸ terms over atom pairs. D3 makes the C6 coefficients depend on each atom's
coordination number; '(BJ)' is Becke-Johnson damping, which keeps a finite attraction at short distance
instead of switching it off (zero damping, D3(0)). D4 makes them depend on atomic charges too. The '-D' of
ωB97X-D is Chai and Head-Gordon's own older pairwise term; VV10 (in ωB97M-V) is a nonlocal functional for the
same effect.

- **Good for:** In QuantaBench: AIMNet2 (D3(BJ)), GFN2-xTB and GFN-FF (D4), GFN1-xTB (D3(BJ)), PM6-D3H4X (D3),
  PM7 (built in), ωB97X-D (-D), B3LYP-D3(BJ) and PBE0-D3(BJ). None in plain B3LYP, HF or r2SCAN as listed;
  M06-2X only at medium range.
- **Watch out:** A functional without one, like plain B3LYP in Spartan's runs, lacks the attraction between
  chain segments that touch, so it tends to favour extended conformers over folded ones ([B3LYP (no
  dispersion)](#b3lyp)). For PFAS ions in water, GFN2/ALPB (D4) and Spartan's B3LYP/C-PCM disagree on folded
  vs extended conformers by tens of kJ/mol (VAL-3). A DFT test found dispersion only a small part of that
  (D3(BJ): about 10 kJ/mol): ωB97X-D and B3LYP-D3(BJ) also put Spartan's pick below GFN2's in most of those
  ions, and GFN2/ALPB is the outlier.
- **Cost:** Negligible: milliseconds even for large molecules.
- **Dispersion:** This entry.
- **Solvation:** Not applicable.
- **See also:** [B3LYP (no dispersion)](#b3lyp), [B3LYP-D3(BJ)](#b3lyp-d3bj), [ωB97X-D](#wb97x-d),
  [GFN2-xTB](#gfn2-xtb)
- **References:**
  - Grimme, Hansen, Brandenburg, Bannwarth. Dispersion-corrected mean-field electronic structure methods.
    Chem. Rev. 116, 5105-5154 (2016).
    [doi:10.1021/acs.chemrev.5b00533](https://doi.org/10.1021/acs.chemrev.5b00533)
  - Grimme, Antony, Ehrlich, Krieg. A consistent and accurate ab initio parametrization of density functional
    dispersion correction (DFT-D) for the 94 elements H-Pu. J. Chem. Phys. 132, 154104 (2010).
    [doi:10.1063/1.3382344](https://doi.org/10.1063/1.3382344)
  - Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme. A generally applicable atomic-charge
    dependent London dispersion correction. J. Chem. Phys. 150, 154122 (2019).
    [doi:10.1063/1.5090222](https://doi.org/10.1063/1.5090222)
- **Also found as:** `d3`, `d3bj`, `d3(bj)`, `d4`, `d2`, `london dispersion`, `-d`, `vv10`, `dispersion
  correction`, `becke-johnson`

<a id="single-point"></a>

### Single point vs optimization

> A single point computes the energy at a fixed geometry; an optimization first moves the atoms to a minimum.

A single point is one SCF at the given coordinates: cheap, and the result belongs to that geometry, not to the
method's own minimum. An optimization repeats energy-and-gradient steps until the forces vanish, so the
geometry and the energy are consistent at one level. 'B3LYP/6-311+G\*\*//MMFF' means a B3LYP single point at
an MMFF geometry; Spartan's water numbers are of that kind.

- **Good for:** Single points: properties at a trusted geometry, or re-ranking many conformers cheaply.
  Optimizations: final structures, and frequencies (which need a minimum at the same level).
- **Watch out:** Ranking conformers by single points at another method's geometries mixes two levels: a
  conformer that would relax a lot at the higher level is ranked too high. In Psi4 with PCM, optimizations run
  in the gas phase (no analytic PCM gradients).
- **Cost:** An optimization costs one energy and gradient per step, often tens of steps for a flexible
  molecule.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [CREST level: GFN2-xTB // GFN-FF](#crest-gfn2--gfnff), [Spartan B3LYP/6-311+G\*\* (C-PCM
  water)](#spartan-b3lyp)
- **Also found as:** `single point`, `optimization`, `optimize`, `energy`, `sp`, `opt`, `//`

<a id="boltzmann"></a>

### Boltzmann weighting

> Weights each conformer by exp(−ΔE/RT): at 298 K a conformer 1.36 kcal/mol up gets a tenth of the weight.

QuantaBench gives each kept conformer the weight w = exp(−ΔE/RT) / Σ exp(−ΔE/RT), with ΔE its refinement
energy above the lowest conformer and T the Temperature setting (298.15 K by default). The Boltzmann-averaged
descriptors (`dipole_debye_boltz` and the other `*_boltz` columns) use these weights. RT is 0.593 kcal/mol
(2.48 kJ/mol) at 298 K, so a conformer 4.1 kcal/mol up gets 1/1000 of the lowest's weight.

- **Good for:** Descriptors of flexible molecules, whose volume, area and dipole change between conformers.
- **Watch out:** The weights use electronic energies at the refinement level (in a solvent, including the ALPB
  or COSMO solvation energy), not free energies with vibrational entropy. An error of 1-2 kcal/mol in a
  relative energy changes a weight by a factor of 5-30, so an average is only as good as the ranking.
  Conformers outside the energy window are dropped before weighting.
- **Cost:** Free.
- **Dispersion:** Not applicable.
- **Solvation:** Not applicable.
- **See also:** [RDKit ETKDGv3 + MMFF94s](#rdkit-etkdg), [CREST iMTD-GC](#crest)
- **Also found as:** `boltzmann`, `boltzmann average`, `weights`, `conformer weights`

<a id="homo-lumo"></a>

### HOMO and LUMO energies

> xTB and PM6/PM7 orbital energies are trend descriptors: use them to rank molecules, not as absolute
> energies.

The HOMO and LUMO are the highest occupied and the lowest unoccupied molecular orbitals. In Hartree-Fock,
−ε\_HOMO approximates the ionization energy (Koopmans' theorem); in DFT, and even more in minimal-basis tight
binding, orbital energies depend strongly on the method. The GFN methods were fitted to geometries,
frequencies and noncovalent energies, not to orbital energies.

- **Good for:** QSAR descriptors within one method and one phase; comparing families of molecules.
- **Watch out:** Methods are offset from each other by electronvolts. In QuantaBench's 300-molecule validation
  GFN2's LUMO sits about 5.5 eV below PM6's and its HOMO 1.1 eV below, so its gap is 4.4 eV smaller, though
  the two correlate (r 0.80-0.88; docs/dev/BACKLOG.md, VAL-3). Use one method for a whole data set. In the gas
  phase the HOMO of a multiply charged anion can lie above zero (unbound).
- **Cost:** Free with any SCF method (xTB, NDDO, DFT).
- **Dispersion:** Not applicable.
- **Solvation:** The solvent shifts orbital energies: compare within one phase.
- **See also:** [GFN2-xTB](#gfn2-xtb), [Spartan PM6](#spartan-pm6), [Hartree-Fock (HF)](#hf)
- **References:**
  - Koopmans. Über die Zuordnung von Wellenfunktionen und Eigenwerten zu den einzelnen Elektronen eines Atoms.
    Physica 1, 104-113 (1934).
    [doi:10.1016/S0031-8914(34)90011-2](https://doi.org/10.1016/S0031-8914%2834%2990011-2)
  - Bannwarth, Ehlert, Grimme. GFN2-xTB, an accurate and broadly parametrized self-consistent tight-binding
    quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. J.
    Chem. Theory Comput. 15, 1652-1671 (2019).
    [doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- **Also found as:** `homo`, `lumo`, `gap`, `homo-lumo gap`, `orbital energies`

<a id="auto"></a>

### Auto (best available)

> Takes the best installed engine for the molecule's elements: AIMNet2, then GFN2-xTB, then GFN1-xTB, then
> MMFF94s.

QuantaBench tries AIMNet2, GFN2-xTB (dxtb, then tblite, then the xtb program) and GFN1-xTB in that order and
takes the first one that is installed and has parameters for every element. With a solvent it tries the
engines that can model it first. When it passes over an engine for a molecule, the notes say why.

- **Good for:** Mixed batches where some molecules have elements AIMNet2 lacks (Sn, Hg, metals).
- **Watch out:** Molecules of one batch can end up at different levels: check the method column of summary.csv
  before comparing energies or dipoles. With a solvent, AIMNet2 and dxtb (gas phase only) move behind tblite
  and xtb.
- **Cost:** That of the engine it takes.
- **Dispersion:** That of the engine it takes (all candidates have one).
- **Solvation:** Prefers an engine with a solvent model when a solvent is set.
- **See also:** [AIMNet2](#aimnet2), [GFN2-xTB](#gfn2-xtb), [GFN1-xTB](#gfn1-xtb), [MMFF94s](#mmff94s)
- **Also found as:** `auto (best available)`, `best available`
