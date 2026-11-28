"""OPLS-AA/PF (FF-1): OPLS-AA for fluorinated liquids and PFAS, from published parameters only.

``md.oplsaa`` types alkanes, perfluoroalkanes and (fluoro)benzenes. This module types a wider set of fluorinated
molecules with the same force field family (OPLS-AA: geometric mixing, 1-4 scaling 0.5/0.5, the Watkins & Jorgensen
2001 perfluoroalkyl types), adding only groups for which a published, OPLS-AA-compatible parameter set exists. It is
the backend ``oplsaa-pf`` of ``md.forcefield`` (``--ff oplsaa-pf``); ``oplsaa`` stays the default.

What it covers (the design note, docs/FORCEFIELDS.md, has the reasons and the validation of each)
---------------------------------------------------------------------------------------------------
* alkanes and perfluoroalkanes, exactly as ``md.oplsaa`` types them (same types, charges, terms: a test checks it);
* semifluorinated chains (a perfluoroalkyl block bonded to an alkyl block): the W&J and OPLS-AA types with the
  junction torsions of Pádua 2002 as CL&P carries them (``CLP_DIHEDRALS``);
* aromatic rings: benzene, fluoro-, difluoro- and hexafluorobenzene carbons, -CF3, -CH3, -CH2- and -Br on a ring
  (OPLS-AA types from GROMACS' oplsaa.ff, several added there for Caleman et al. 2012);
* monoalkyl fluorides RCH2F, R2CHF, R3CF (OPLS-AA opls_956-960, which GROMACS marks "tentative");
* alcohols: hydrocarbon alcohols (OPLS-AA), 2,2,2-trifluoroethanol (OPLS-AA, opls_160-165) and the
  1H,1H-perfluoroalcohols CF3(CF2)nCH2OH by Silva et al. 2016's recipe (the TFE parameters for -CF2CH2OH, the
  fluorinated carbon's charge adjusted for neutrality: ``SILVA_RULE``);
* perfluoroalkanesulfonate anions (CF3SO3-, CnF2n+1SO3-) and trifluoroacetate from CL&P (Canongia Lopes & Pádua),
  and the OPLS-AA (Åqvist) cations Li+, Na+, K+ as counter-ions. Liquid slabs are for neutral molecules only.

Everything else is refused with the atom and the reason (``PFTypingError``), never typed by guess: CHF2 and other
partly fluorinated carbons, gem-difluorides, CF2 bonded to C=O, ether O, N or S(=O)2N, chlorine, ... (the coverage
report lists them).

Where every number comes from
-----------------------------
Every table entry keeps the values as its source writes them, in its units, with the line it comes from:

* ``gromacs``: GROMACS ``share/top/oplsaa.ff`` (ffnonbonded.itp / ffbonded.itp) at ``md.oplsaa.GROMACS_COMMIT``,
  the same files ``md.oplsaa`` cites (nm, kJ/mol, kJ mol-1 nm-2, kJ mol-1 rad-2, degrees);
* ``clp``: CL&P's ``il.ff`` at ``CLP_COMMIT`` (Å, kJ/mol; bond and angle constants are k/2 as in GROMACS, so
  K_LAMMPS = k / 2 / 4.184; torsions are OPLS V1..V4 in kJ/mol).

``scripts/check_oplsaa_pf_params.py`` re-reads both files and checks every entry, value and line number.

Conventions: no charge is redistributed. A molecule's charges are the sum of its atom types' published charges
(OPLS-AA's transferable neutral groups), with one derived value (``SILVA_RULE``), and must add up to the formal
charge exactly, or the molecule is refused. Each term is looked up in a fixed order (``term_parameters``); a term
found nowhere refuses the molecule. Which published recipe a molecule follows, or that it combines groups no paper
combined, is reported (``PFAssignment.recipe``, ``.tier``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from . import oplsaa as O

KJ_PER_KCAL = O.KJ_PER_KCAL

CLP_COMMIT = "898c19d5a4622ae60af727a38aa76e2fd34f18e7"
CLP_URL = f"https://github.com/paduagroup/clandp/blob/{CLP_COMMIT}/il.ff"

REFERENCES = {
    **{k: v for k, v in O.REFERENCES.items() if k in ("jorgensen1996", "watkins2001", "jorgensen2004", "gromacs")},
    "caleman2012": "Caleman, van Maaren, Hong, Hub, Costa, van der Spoel. Force field benchmark of organic liquids. "
                   "J. Chem. Theory Comput. 2012, 8, 61-74. doi:10.1021/ct200731v (the aromatic F/CF3/Br types "
                   "opls_720-730, as foyer's oplsaa.xml cites them)",
    "padua2002": "Pádua. Torsion energy profiles and force fields derived from ab initio calculations for simulations "
                 "of hydrocarbon-fluorocarbon diblocks and perfluorinated n-alkanes. J. Phys. Chem. A 2002, 106, "
                 "10116-10123. doi:10.1021/jp025732n (the junction torsions, as CL&P il.ff carries them)",
    "clp2004": "Canongia Lopes, Pádua. Molecular force field for ionic liquids composed of triflate or "
               "bistriflylimide anions. J. Phys. Chem. B 2004, 108, 16893-16898. doi:10.1021/jp0476545",
    "shimizu2010": "Shimizu, Almantariotis, Costa Gomes, Pádua, Canongia Lopes. Molecular force field for ionic "
                   "liquids V. J. Phys. Chem. B 2010, 114, 3592-3600. doi:10.1021/jp9120468 (perfluoroalkane"
                   "sulfonylamides; CL&P il.ff 'longer perfluoroalkanesulfonylamides')",
    "goloviznina2017": "Goloviznina et al. (CL&P il.ff cites 'PCCP 43 (2017) 29617' for triflate's O, OTF)",
    "clp_tfa": "CL&P il.ff 'trifluorocetate PCCP 17 (2015) 22321' (trifluoroacetate anion)",
    "aqvist1990": "Åqvist. Ion-water interaction potentials derived from free energy perturbation simulations. "
                  "J. Phys. Chem. 1990, 94, 8021-8024. doi:10.1021/j100384a009 (OPLS-AA opls_406-408)",
    "silva2016": "Silva, Morgado, Haley, Trejos Montoya, McCabe, Martins, Filipe. Vapor pressure and liquid density of "
                 "fluorinated alcohols: experimental, simulation and GC-SAFT-VR predictions. Fluid Phase Equilib. "
                 "2016, 425, 297-304. doi:10.1016/j.fluid.2016.06.011 (W&J tail + the TFE parameters for -CF2CH2OH, "
                 "the fluorinated carbon's charge adjusted for neutrality, Pádua's junction torsions)",
    "clp": f"CL&P force field il.ff (Canongia Lopes, Pádua et al.), MIT, commit {CLP_COMMIT[:10]}: {CLP_URL}",
}

MIXING = O.MIXING
SPECIAL_LJ = O.SPECIAL_LJ
SPECIAL_COUL = O.SPECIAL_COUL


class PFTypingError(ValueError):
    """The molecule has an atom or a term that no published source here covers."""


# ------------------------------------------------------------------------------------------------ atom types
@dataclass(frozen=True)
class PFType:
    """One atom type: LJ and charge in the source's units, the GROMACS bond type and the CL&P class used to look up
    bonded terms, its family (for the recipe) and its source line."""

    name: str
    element: str
    mass: float
    charge: float | None         # e; None: derived per molecule (SILVA_RULE)
    sigma_A: float
    epsilon_kj: float
    btype: str                   # GROMACS bond type ("" = none)
    clp: str                     # CL&P class ("" = none)
    family: str
    description: str
    source: str                  # "gromacs ffnonbonded.itp line N" / "clp il.ff line N"
    reference: str               # REFERENCES key

    @property
    def epsilon_kcal(self) -> float:
        return self.epsilon_kj / KJ_PER_KCAL

    def cite(self) -> str:
        return f"{self.name} ({self.description}): {self.source}; [{self.reference}]"


def _g(name, el, mass, q, sig_nm, eps_kj, bt, clp, fam, desc, line, ref) -> PFType:
    return PFType(name, el, mass, q, sig_nm * 10.0, eps_kj, bt, clp, fam, desc,
                  f"GROMACS oplsaa.ff/ffnonbonded.itp line {line}", ref)


def _c(name, el, mass, q, sig_A, eps_kj, clp, fam, desc, line, ref, bt="") -> PFType:
    return PFType(f"clp_{name}", el, mass, q, sig_A, eps_kj, bt, clp, fam, desc, f"CL&P il.ff line {line}", ref)


TYPES: dict[str, PFType] = {t.name: t for t in (
    # alkanes, perfluoroalkanes, benzene, fluorobenzene: the md.oplsaa table (same lines)
    *(_g(t.name, t.element, t.mass, t.charge, t.sigma_nm, t.epsilon_kj, t.bond_type,
         {"opls_140": "HC", "opls_145": "CA", "opls_146": "HA", "opls_718": "CA", "opls_719": "F",
          "opls_965": "F"}.get(t.name, "CF" if t.name in O.PERFLUORO_CARBONS else "CT"),
         "perfluoroalkane" if t.name in O.PERFLUORO_CARBONS or t.name == "opls_965" else
         ("aromatic" if t.name in ("opls_145", "opls_146", "opls_718", "opls_719") else "alkane"),
         t.description, t.line, t.reference) for t in O.ATOM_TYPES.values()),
    # aromatic substituents
    _g("opls_148", "C", 12.011, -0.065, 0.350, 0.276144, "CT", "CT", "aromatic", "CH3 on a benzene ring (toluene)",
       159, "jorgensen1996"),
    _g("opls_149", "C", 12.011, -0.005, 0.350, 0.276144, "CT", "CT", "aromatic", "CH2 on a benzene ring "
       "(ethylbenzene)", 160, "jorgensen1996"),
    _g("opls_720", "C", 12.011, 0.130, 0.355, 0.292880, "CA", "CA", "aromatic", "ring C(F) between two C(F) "
       "(hexafluorobenzene)", 702, "caleman2012"),
    _g("opls_721", "F", 18.9984, -0.130, 0.285, 0.255224, "F", "F", "aromatic", "F on opls_720 (hexafluorobenzene)",
       703, "caleman2012"),
    _g("opls_724", "C", 12.011, 0.150, 0.355, 0.292880, "CA", "CA", "aromatic", "ring C bonded to CF3 "
       "(trifluoromethylbenzene)", 705, "caleman2012"),
    _g("opls_725", "C", 12.011, 0.450, 0.325, 0.259408, "CT", "CF", "aromatic", "CF3 on a benzene ring "
       "(trifluoromethylbenzene)", 706, "caleman2012"),
    _g("opls_726", "F", 18.9984, -0.200, 0.294, 0.255224, "F", "F", "aromatic", "F of a ring CF3 "
       "(trifluoromethylbenzene)", 707, "caleman2012"),
    _g("opls_727", "C", 12.011, 0.200, 0.355, 0.292880, "CA", "CA", "aromatic", "ring C(F) next to one C(F) "
       "(difluorobenzenes)", 708, "caleman2012"),
    _g("opls_728", "F", 18.9984, -0.200, 0.285, 0.255224, "F", "F", "aromatic", "F on opls_727 (difluorobenzenes)",
       709, "caleman2012"),
    _g("opls_729", "C", 12.011, 0.200, 0.355, 0.292880, "CA", "CA", "aromatic", "ring C(Br) (bromobenzene)", 710,
       "caleman2012"),
    _g("opls_730", "Br", 79.904, -0.200, 0.347, 1.96648, "Br", "", "aromatic", "Br on a benzene ring "
       "(bromobenzene)", 711, "caleman2012"),
    # monoalkyl fluorides (GROMACS ffoplsaa.atp: "tentative")
    _g("opls_956", "F", 18.9984, -0.220, 0.294, 0.255224, "F", "", "alkyl_fluoride", "F in monoalkyl fluorides "
       "(tentative)", 810, "jorgensen1996"),
    _g("opls_957", "C", 12.011, 0.020, 0.350, 0.276144, "CT", "", "alkyl_fluoride", "RCH2F in monoalkyl fluorides "
       "(tentative)", 811, "jorgensen1996"),
    _g("opls_958", "H", 1.008, 0.100, 0.250, 0.125520, "HC", "", "alkyl_fluoride", "H in RCHF in monoalkyl "
       "fluorides (tentative)", 812, "jorgensen1996"),
    _g("opls_959", "C", 12.011, 0.120, 0.350, 0.276144, "CT", "", "alkyl_fluoride", "R2CHF in monoalkyl fluorides "
       "(tentative)", 813, "jorgensen1996"),
    _g("opls_960", "C", 12.011, 0.220, 0.350, 0.276144, "CT", "", "alkyl_fluoride", "R3CF in monoalkyl fluorides "
       "(tentative)", 814, "jorgensen1996"),
    # alcohols
    _g("opls_154", "O", 15.9994, -0.683, 0.312, 0.711280, "OH", "OH", "alcohol", "O of a mono alcohol", 165,
       "jorgensen1996"),
    _g("opls_155", "H", 1.008, 0.418, 0.0, 0.0, "HO", "HO", "alcohol", "H(O) of a mono alcohol", 166,
       "jorgensen1996"),
    _g("opls_157", "C", 12.011, 0.145, 0.350, 0.276144, "CT", "CT", "alcohol", "C of CH3/CH2 alcohols (C-OH)", 168,
       "jorgensen1996"),
    _g("opls_158", "C", 12.011, 0.205, 0.350, 0.276144, "CT", "CT", "alcohol", "C of CH alcohols (C-OH)", 169,
       "jorgensen1996"),
    _g("opls_159", "C", 12.011, 0.265, 0.350, 0.276144, "CT", "CT", "alcohol", "C of C alcohols (C-OH)", 170,
       "jorgensen1996"),
    # 2,2,2-trifluoroethanol (the -CF2CH2OH segment of Silva et al. 2016's 1H,1H-perfluoroalcohols)
    _g("opls_160", "C", 12.011, 0.126, 0.350, 0.276144, "CT_4", "CT", "tfe", "CH2 of trifluoroethanol", 171,
       "silva2016"),
    _g("opls_161", "C", 12.011, 0.532, 0.325, 0.259408, "CT", "CF", "tfe", "CF3 carbon of trifluoroethanol", 172,
       "silva2016"),
    _g("opls_162", "O", 15.9994, -0.635, 0.307, 0.711280, "OH", "OH", "tfe", "OH of trifluoroethanol", 173,
       "silva2016"),
    _g("opls_163", "H", 1.008, 0.429, 0.0, 0.0, "HO", "HO", "tfe", "HO of trifluoroethanol", 174, "silva2016"),
    _g("opls_164", "F", 18.9984, -0.206, 0.294, 0.255224, "F", "F", "tfe", "F of trifluoroethanol", 175,
       "silva2016"),
    _g("opls_165", "H", 1.008, 0.083, 0.250, 0.125520, "HC", "HC", "tfe", "H(C) of trifluoroethanol", 176,
       "silva2016"),
    # counter-ions: OPLS-AA's Åqvist cations
    _g("opls_406", "Li", 6.941, 1.0, 0.212645, 0.0764793, "", "", "ion", "Li+ (Åqvist)", 422, "aqvist1990"),
    _g("opls_407", "Na", 22.98977, 1.0, 0.333045, 0.0115980, "", "", "ion", "Na+ (Åqvist)", 423, "aqvist1990"),
    _g("opls_408", "K", 39.0983, 1.0, 0.493463, 0.00137235, "", "", "ion", "K+ (Åqvist)", 424, "aqvist1990"),
    # CL&P anion heads (the bond type CT/F lets a chain term next to the head use the GROMACS tables)
    _c("CBT", "C", 12.011, 0.35, 3.50, 0.27614, "CF", "sulfonate", "CF3 of triflate", 133, "clp2004", bt="CT"),
    _c("SBT", "S", 32.066, 1.02, 3.55, 1.04600, "SB", "sulfonate", "S of triflate / sulfonyl", 134, "clp2004"),
    _c("F1", "F", 18.998, -0.16, 3.118, 0.25540, "FB", "sulfonate", "F on the carbon next to S", 137, "clp2004",
       bt="F"),
    _c("OTF", "O", 15.999, -0.63, 3.15, 0.83736, "OB", "sulfonate", "O of triflate's SO3", 139, "goloviznina2017"),
    _c("C1F", "C", 12.011, 0.19, 3.50, 0.27614, "CF", "sulfonate", "CF2 next to S (perfluoroalkanesulfonyl)", 141,
       "shimizu2010", bt="CT"),
    _c("O2F", "O", 15.999, -0.75, 2.96, 0.87864, "O2", "carboxylate", "O of trifluoroacetate", 165, "clp_tfa"),
    _c("CO2", "C", 12.011, 0.70, 3.75, 0.43932, "CO", "carboxylate", "carboxylate C", 151, "clp_tfa"),
    _c("CFA", "C", 12.011, 0.40, 3.50, 0.27614, "CF", "carboxylate", "CF3 of trifluoroacetate", 166, "clp_tfa"),
    _c("FFA", "F", 18.998, -0.20, 2.95, 0.22175, "F", "carboxylate", "F of trifluoroacetate", 167, "clp_tfa"),
)}

#: Silva et al. 2016, Sec. 2.4: "the -CF2CH2OH segment was modeled with the parameters developed by Duffy for
#: trifluoroethanol, adjusting the partial charge of the fluorinated carbon to maintain the neutrality of the
#: molecule". Applied here to a CF2 (opls_161's LJ, its two F opls_164) bonded to the TFE CH2: its charge is minus
#: the sum of the rest of the -CF2CH2OH segment (a W&J tail is neutral by itself): 0.326 e.
SILVA_RULE = ("derived by Silva et al. 2016's rule (TFE parameters for -CF2CH2OH, the fluorinated carbon's charge "
              "set so the segment is neutral)")
SILVA_CF2 = "silva_161"
TYPES[SILVA_CF2] = PFType(SILVA_CF2, "C", 12.011, None, 3.25, 0.259408, "CT", "CF", "tfe",
                          "CF2 of a 1H,1H-perfluoroalcohol's -CF2CH2OH (opls_161's LJ)",
                          "GROMACS oplsaa.ff/ffnonbonded.itp line 172 (LJ); charge " + SILVA_RULE, "silva2016")

PERFLUORO_CARBONS = O.PERFLUORO_CARBONS
AROMATIC_C = frozenset({"opls_145", "opls_718", "opls_720", "opls_724", "opls_727", "opls_729"})
ION_TYPES = {"Li": "opls_406", "Na": "opls_407", "K": "opls_408"}

# ------------------------------------------------------------------------------------------- bonded tables
#: GROMACS ffbonded.itp [ bondtypes ] funct 1: (b0 nm, kb kJ mol-1 nm-2, line, comment); V = ½ kb (b - b0)²
GMX_BONDS: dict[tuple[str, str], tuple[float, float, int, str]] = {
    ("CA", "CA"): (0.14000, 392459.2, 97, "TRP,TYR,PHE"),
    ("CA", "CT"): (0.15100, 265265.6, 113, "PHE,TYR"),
    ("CA", "HA"): (0.10800, 307105.6, 140, "PHE, etc."),
    ("CT", "CT"): (0.15290, 224262.4, 192, "CHARMM 22 parameter file"),
    ("CT", "CT_4"): (0.15290, 224262.4, 195, "Trifluoroethanol"),
    ("CT", "HC"): (0.10900, 284512.0, 196, "CHARMM 22 parameter file"),
    ("CT_4", "HC"): (0.10900, 284512.0, 199, "Trifluoroethanol"),
    ("CT", "OH"): (0.14100, 267776.0, 213, ""),
    ("CT_4", "OH"): (0.14100, 267776.0, 214, "ifluoroethanol"),
    ("CA", "Br"): (0.18700, 251040.0, 233, "wlj"),
    ("HO", "OH"): (0.09450, 462750.4, 259, "SUG(OL) wlj mod 0.96-> 0.945"),
    ("CT", "F"): (0.13320, 307105.6, 276, "PAK CT-F for CHF3 (emd 5-09-94)"),
    ("CA", "F"): (0.13540, 351456.0, 277, "wlj"),
}

#: [ angletypes ] funct 1: (θ0 degrees, k kJ mol-1 rad-2, line, comment); V = ½ k (θ - θ0)²
GMX_ANGLES: dict[tuple[str, str, str], tuple[float, float, int, str]] = {
    ("CA", "CA", "Br"): (120.000, 627.6, 524, "wlj"),
    ("CA", "CA", "F"): (120.000, 669.44, 530, "wlj"),
    ("CA", "CA", "CA"): (120.000, 527.184, 587, "PHE(OL)"),
    ("CA", "CA", "CT"): (120.000, 585.76, 601, "PHE(OL)"),
    ("CA", "CA", "HA"): (120.000, 292.88, 603, ""),
    ("CA", "CT", "CT"): (114.000, 527.184, 862, "PHE(OL)         SCH JPC  79,2379"),
    ("CA", "CT", "HC"): (109.500, 292.88, 864, ""),
    ("CT", "CT", "CT"): (112.700, 488.273, 873, "CHARMM 22 parameter file"),
    ("CT", "CT", "HC"): (110.700, 313.8, 881, "CHARMM 22 parameter file"),
    ("CT", "CT_4", "HC"): (110.700, 313.8, 884, "Trifluoroethanol"),
    ("CT", "CT", "OH"): (109.500, 418.4, 909, ""),
    ("CT", "CT_4", "OH"): (109.500, 418.4, 911, "Trifluoroethanol"),
    ("CA", "CT", "OH"): (109.500, 418.4, 912, "wlj"),
    ("HC", "CT", "HC"): (107.800, 276.144, 929, "CHARMM 22 parameter file"),
    ("HC", "CT_4", "HC"): (107.800, 276.144, 932, "Trifluoroethanol"),
    ("HC", "CT", "OH"): (109.500, 292.88, 961, ""),
    ("HC", "CT_4", "OH"): (109.500, 292.88, 965, "fluoroethanol"),
    ("CT", "OH", "HO"): (108.500, 460.24, 1095, ""),
    ("CT_4", "OH", "HO"): (108.500, 460.24, 1096, "Trifluoroethanol"),
    ("CA", "CT", "CA"): (109.500, 334.72, 1167, "-idem-"),
    ("CT", "CT", "F"): (109.500, 418.4, 1187, "PAK F-CT-HC (emd 5-09-94)"),
    ("CT_4", "CT", "F"): (109.500, 418.4, 1188, "Trifluoroethanol"),
    ("CA", "CT", "F"): (109.500, 418.4, 1189, "wlj"),
    ("F", "CT", "F"): (109.100, 644.336, 1190, "PAK F-CT-F  (emd 5-09-94)"),
    ("HC", "CT", "F"): (107.000, 334.72, 1191, "wlj"),
}

#: [ dihedraltypes ] funct 3, Ryckaert-Bellemans C0..C5 (kJ/mol); "X" = any
GMX_DIHEDRALS: dict[tuple[str, str, str, str], tuple[tuple[float, ...], int, str]] = {
    ("CA", "CA", "CT", "F"): ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1405, "fluoromethyl benzene"),
    ("CA", "CA", "CT", "CT"): ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1406, "ethyl benzene"),
    ("CA", "CA", "CT", "HC"): ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1408, "ethyl benzene"),
    ("CA", "CT", "CT", "CT"): ((2.9288, -1.4644, 0.2092, -1.6736, 0.0, 0.0), 1432, ""),
    ("CA", "CT", "CT", "HC"): ((0.9665, 2.89951, 0.0, -3.86601, 0.0, 0.0), 1433, "ethyl benzene"),
    ("CA", "CT", "CT", "OH"): ((2.87441, 0.58158, 2.092, -5.54799, 0.0, 0.0), 1439, "from alcohol"),
    ("CA", "CT", "OH", "HO"): ((-1.8828, 1.8828, 0.0, 0.0, 0.0, 0.0), 1444, "benzyl alcohols"),
    ("CT", "CT", "CT", "F"): ((1.4644, 1.8828, 0.0, -3.3472, 0.0, 0.0), 1593, "alkyl fluoride"),
    ("CT", "CT", "CT", "CT"): ((2.9288, -1.4644, 0.2092, -1.6736, 0.0, 0.0), 1596, "hydrocarbon all-atom"),
    ("CT", "CT", "CT", "HC"): ((0.6276, 1.8828, 0.0, -2.5104, 0.0, 0.0), 1599, "hydrocarbon all-atom"),
    ("CT", "CT", "CT", "OH"): ((2.87441, 0.58158, 2.092, -5.54799, 0.0, 0.0), 1607, "alcohols, ethers AA"),
    ("CT", "CT", "OH", "HO"): ((-0.4435, 3.83255, 0.72801, -4.11705, 0.0, 0.0), 1630, "alcohols AA"),
    ("CT", "CT_4", "OH", "HO"): ((0.26778, -9.36798, 9.1002, 0.0, 0.0, 0.0), 1631, "trifluoroethanol"),
    ("F", "CT", "CT", "F"): ((-5.23, 5.23, 0.0, 0.0, 0.0, 0.0), 1756, "1,2-difluoride"),
    ("F", "CT", "CT", "HC"): ((0.8368, 2.5104, 0.0, -3.3472, 0.0, 0.0), 1757, "alkyl fluoride"),
    ("F", "CT", "CT", "OH"): ((1.12968, 3.38904, 0.0, -4.51872, 0.0, 0.0), 1758, "trifluoroethanol"),
    ("F", "CT", "CT_4", "HC"): ((0.65689, 1.97066, 0.0, -2.62755, 0.0, 0.0), 1759, "trifluoroethanol"),
    ("F", "CT", "CT_4", "OH"): ((1.12968, 3.38904, 0.0, -4.51872, 0.0, 0.0), 1760, "trifluoroethanol"),
    ("HC", "CT", "CT", "HC"): ((0.6276, 1.8828, 0.0, -2.5104, 0.0, 0.0), 1824, "hydrocarbon *new* 11/99"),
    ("HC", "CT", "CT", "OH"): ((0.97905, 2.93716, 0.0, -3.91622, 0.0, 0.0), 1831, "alcohols, ethers AA"),
    ("HC", "CT", "OH", "HO"): ((0.9414, 2.8242, 0.0, -3.7656, 0.0, 0.0), 1842, "alcohols AA"),
    ("HC", "CT_4", "OH", "HO"): ((0.99579, 2.98738, 0.0, -3.98316, 0.0, 0.0), 1843, "trifluoroethanol"),
    ("OH", "CT", "CT", "OH"): ((18.96607, -18.96607, 0.0, 0.0, 0.0, 0.0), 1907, "hexopyranoses"),
    ("CA", "CA", "CT", "X"): ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1912, "ethyl benzene"),
    ("X", "CA", "CA", "X"): ((30.334, 0.0, -30.334, 0.0, 0.0, 0.0), 1956, "aromatic ring"),
    ("CA", "CA", "CT", "OH"): ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 2012, "(From wildcard) benzyl alcohols & ethers"),
}

#: CL&P il.ff BONDS: (r0 Å, k kJ mol-1 Å-2 in k/2 form, line)
CLP_BONDS: dict[tuple[str, str], tuple[float, float, int]] = {
    ("FB", "CF"): (1.323, 3698.0, 243),
    ("CF", "SB"): (1.818, 1950.0, 244),
    ("SB", "OB"): (1.437, 5331.0, 245),
    ("CO", "O2"): (1.250, 5489.0, 253),
    ("CF", "CO"): (1.522, 2653.0, 258),
    ("F", "CF"): (1.332, 3071.1, 223),
    ("CF", "CF"): (1.529, 2242.6, 224),
}

#: CL&P il.ff ANGLES: (θ0 degrees, k kJ mol-1 rad-2 in k/2 form, line)
CLP_ANGLES: dict[tuple[str, str, str], tuple[float, float, int]] = {
    # fluoroalkyl / junction (Watkins & Jorgensen 2001, Pádua 2002)
    ("HC", "CT", "CF"): (110.7, 313.8, 316),
    ("F", "CF", "CT"): (109.5, 418.4, 317),
    ("CT", "CF", "CF"): (112.7, 488.3, 318),
    ("CT", "CT", "CF"): (112.7, 488.3, 319),
    ("F", "CF", "F"): (109.1, 644.3, 320),
    ("F", "CF", "CF"): (109.5, 418.4, 321),
    ("CF", "CF", "CF"): (112.7, 488.3, 322),
    # triflate, perfluoroalkanesulfonyl
    ("FB", "CF", "FB"): (107.1, 781.0, 351),
    ("FB", "CF", "SB"): (111.7, 694.0, 352),
    ("OB", "SB", "OB"): (118.5, 969.0, 353),
    ("CF", "SB", "OB"): (102.6, 870.0, 354),
    ("SB", "CF", "CF"): (115.9, 418.4, 359),
    ("FB", "CF", "CF"): (109.5, 418.4, 360),
    # carboxylates, trifluoroacetate
    ("O2", "CO", "O2"): (126.0, 669.4, 368),
    ("F", "CF", "CO"): (109.5, 418.4, 378),
    ("CF", "CO", "O2"): (117.0, 585.8, 379),
}

#: CL&P il.ff DIHEDRALS (OPLS V1..V4, kJ/mol, line)
CLP_DIHEDRALS: dict[tuple[str, str, str, str], tuple[tuple[float, float, float, float], int]] = {
    # fluoroalkyl / hydrocarbon junction: Pádua 2002 (il.ff "fluoroalkyl JPCA 105 (2001) 4118; JPCA 106 (2002) 10116")
    ("F", "CF", "CT", "HC"): ((0.0, 0.0, 1.2134, 0.0), 455),
    ("F", "CF", "CT", "CT"): ((0.0, 0.0, 1.9372, 0.0), 456),
    ("F", "CF", "CF", "CT"): ((0.0, 0.0, 2.7656, 0.0), 457),
    ("HC", "CT", "CT", "CF"): ((0.0, 0.0, 0.5565, 0.0), 458),
    ("HC", "CT", "CF", "CF"): ((0.0, 0.0, 0.7573, 0.0), 459),
    ("CF", "CF", "CF", "CT"): ((14.6750, -0.9179, -2.8980, -2.0174), 460),
    ("CF", "CF", "CT", "CT"): ((-0.5881, -0.7642, -0.3170, -0.3179), 461),
    # perfluoroalkyl chain (W&J 2001 as CL&P writes it; used only for terms that touch a CL&P head atom)
    ("F", "CF", "CF", "F"): ((-10.4600, 0.0, 1.0460, 0.0), 463),
    ("F", "CF", "CF", "CF"): ((1.2552, 0.0, 1.6736, 0.0), 464),
    ("CF", "CF", "CF", "CF"): ((27.7064, 3.9664, -5.8074, -8.8617), 465),
    # triflate, perfluoroalkanesulfonyl
    ("OB", "SB", "CF", "FB"): ((0.0, 0.0, 1.4510, 0.0), 504),
    ("F", "CF", "CF", "FB"): ((-10.4600, 0.0, 1.0460, 0.0), 509),
    ("SB", "CF", "CF", "FB"): ((0.0, 0.0, 1.4530, 0.0), 510),
    ("SB", "CF", "CF", "F"): ((0.0, 0.0, 1.4530, 0.0), 511),
    ("OB", "SB", "CF", "CF"): ((0.0, 0.0, -0.7400, 0.0), 512),
    ("SB", "CF", "CF", "CF"): ((50.0900, 0.0, -4.6260, -4.0080), 514),
    # trifluoroacetate
    ("F", "CF", "CO", "O2"): ((0.0, 0.0, 0.0, 0.0), 533),
}

#: CL&P il.ff IMPROPER (central atom third; OPLS V1..V4, kJ/mol, line)
CLP_IMPROPERS: dict[tuple[str, str, str, str], tuple[tuple[float, float, float, float], int]] = {
    ("CF", "O2", "CO", "O2"): ((0.0, 87.8640, 0.0, 0.0), 572),
}


# --------------------------------------------------------------------------------------------- the typing
@dataclass
class PFAssignment:
    types: list[str]
    charges: list[float]
    charge_notes: dict[int, str] = field(default_factory=dict)     # atom index -> how its charge was derived
    families: set[str] = field(default_factory=set)
    junction: bool = False
    notes: list[str] = field(default_factory=list)


def _refuse(atom: Chem.Atom, why: str) -> PFTypingError:
    return PFTypingError(f"atom {atom.GetIdx()} ({atom.GetSymbol()}) {why}")


def _benzene_ring(atom: Chem.Atom) -> tuple[int, ...] | None:
    """The six-membered all-carbon aromatic ring of ``atom``, if it is in exactly one ring and that ring is one."""
    mol = atom.GetOwningMol()
    rings = [r for r in mol.GetRingInfo().AtomRings() if atom.GetIdx() in r]
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    r = rings[0]
    if all(mol.GetAtomWithIdx(i).GetSymbol() == "C" and mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
        return r
    return None


def _counts(atom: Chem.Atom) -> dict[str, int]:
    c: dict[str, int] = {}
    for n in atom.GetNeighbors():
        c[n.GetSymbol()] = c.get(n.GetSymbol(), 0) + 1
    return c


def _is_sp3_c(a: Chem.Atom) -> bool:
    return a.GetSymbol() == "C" and not a.GetIsAromatic() and a.GetDegree() == 4 and a.GetFormalCharge() == 0


def _n(a: Chem.Atom, sym: str) -> int:
    return sum(1 for n in a.GetNeighbors() if n.GetSymbol() == sym)


def _perfluoro_c(a: Chem.Atom) -> bool:
    """sp3 carbon with no H and at least one F, all other neighbours carbon or the heads handled here (S, C(=O)O-)."""
    return _is_sp3_c(a) and _n(a, "H") == 0 and _n(a, "F") >= 1 and all(
        n.GetSymbol() in ("C", "F", "S", "O") for n in a.GetNeighbors()) and _n(a, "O") == 0


def _hydro_c(a: Chem.Atom) -> bool:
    return _is_sp3_c(a) and _n(a, "F") == 0 and all(n.GetSymbol() in ("C", "H", "O") for n in a.GetNeighbors())


def _sulfonate_s(a: Chem.Atom) -> bool:
    if a.GetSymbol() != "S" or a.GetDegree() != 4:
        return False
    os_ = [n for n in a.GetNeighbors() if n.GetSymbol() == "O"]
    cs = [n for n in a.GetNeighbors() if n.GetSymbol() == "C"]
    return (len(os_) == 3 and len(cs) == 1 and all(o.GetDegree() == 1 for o in os_)
            and sum(o.GetFormalCharge() for o in os_) == -1)


def _carboxylate_c(a: Chem.Atom) -> bool:
    if a.GetSymbol() != "C" or a.GetDegree() != 3:
        return False
    os_ = [n for n in a.GetNeighbors() if n.GetSymbol() == "O"]
    return len(os_) == 2 and all(o.GetDegree() == 1 for o in os_) and sum(o.GetFormalCharge() for o in os_) == -1


def assign(mol: Chem.Mol) -> PFAssignment:
    """OPLS-AA/PF type and charge of every atom of ``mol`` (one molecule or ion, explicit hydrogens), or
    ``PFTypingError`` naming the first atom no source covers."""
    if any(a.GetTotalNumHs(includeNeighbors=False) > 0 for a in mol.GetAtoms() if a.GetSymbol() != "H"):
        raise PFTypingError("the molecule needs explicit hydrogens (Chem.AddHs)")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise PFTypingError("one molecule or ion at a time (the SMILES has several fragments)")
    n = mol.GetNumAtoms()
    types: list[str] = [""] * n
    notes: list[str] = []
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons():
            raise _refuse(a, "is a radical")
        if a.GetSymbol() not in ("C", "H", "F", "O", "S", "Br", "Li", "Na", "K"):
            raise _refuse(a, "is not C, H, F, O, S or Br (no published OPLS-AA fluorinated-group parameters "
                             "for it are in this table)")
    for a in mol.GetAtoms():
        if a.GetSymbol() == "C" and _is_sp3_c(a) and _n(a, "H") and _n(a, "F"):
            carbons = [x for x in a.GetNeighbors() if x.GetSymbol() == "C"]
            mono = (_n(a, "F") == 1 and carbons and all(_hydro_c(x) and not x.GetIsAromatic() for x in carbons)
                    and all(x.GetSymbol() in ("C", "H", "F") for x in a.GetNeighbors()))
            if not mono:
                what = {2: "CHF2", 3: "?"}.get(_n(a, "F"), "CH2F/CHF next to a fluorinated, aromatic or hetero atom, "
                                                         "or CH3F")
                raise _refuse(a, f"is a partly fluorinated carbon ({what}): no published OPLS-AA type (CHF2 groups, "
                                 "as in omega-H telomers, HFCs and HFEs, need their own parameters)")
    if n == 1:
        a = mol.GetAtomWithIdx(0)
        if a.GetSymbol() in ION_TYPES and a.GetFormalCharge() == 1:
            t = ION_TYPES[a.GetSymbol()]
            return PFAssignment([t], [1.0], families={"ion"})
        raise _refuse(a, "is a single atom that is not Li+, Na+ or K+")

    # 1. anion heads (CL&P): sulfonate on a fluorinated carbon, carboxylate on CF3
    for a in mol.GetAtoms():
        if a.GetSymbol() == "S":
            if not _sulfonate_s(a):
                raise _refuse(a, "is S outside a sulfonate anion (sulfonyl fluorides, sulfonamides, sulfonic acids "
                                 "and sulfones have no published OPLS-AA fluorinated parameters here)")
            c = next(x for x in a.GetNeighbors() if x.GetSymbol() == "C")
            if not (_is_sp3_c(c) and _n(c, "H") == 0 and _n(c, "F") in (2, 3)):
                raise _refuse(a, "is a sulfonate whose carbon is not CF3 or CF2 (CL&P covers CF3SO3- and "
                                 "perfluoroalkanesulfonyl only)")
            types[a.GetIdx()] = "clp_SBT"
            for o in a.GetNeighbors():
                if o.GetSymbol() == "O":
                    types[o.GetIdx()] = "clp_OTF"
            if _n(c, "F") == 3:
                types[c.GetIdx()] = "clp_CBT"
            else:
                other = [x for x in c.GetNeighbors() if x.GetSymbol() == "C"]
                if len(other) != 1 or not _perfluoro_c(other[0]):
                    raise _refuse(c, "is the CF2 of a sulfonate not bonded to a perfluoroalkyl carbon")
                types[c.GetIdx()] = "clp_C1F"
            for f in c.GetNeighbors():
                if f.GetSymbol() == "F":
                    types[f.GetIdx()] = "clp_F1"
        elif a.GetSymbol() == "C" and _carboxylate_c(a):
            c = next(x for x in a.GetNeighbors() if x.GetSymbol() == "C") if _n(a, "C") == 1 else None
            if c is None or not (_is_sp3_c(c) and _n(c, "F") == 3):
                raise _refuse(a, "is a carboxylate not on a CF3 group (CL&P has trifluoroacetate only; a longer "
                                 "perfluoroalkanoate's alpha-CF2 has no published type)")
            types[a.GetIdx()] = "clp_CO2"
            for o in a.GetNeighbors():
                if o.GetSymbol() == "O":
                    types[o.GetIdx()] = "clp_O2F"
            types[c.GetIdx()] = "clp_CFA"
            for f in c.GetNeighbors():
                if f.GetSymbol() == "F":
                    types[f.GetIdx()] = "clp_FFA"

    # 2. alcohols: hydrocarbon (OPLS-AA) or the TFE / Silva -CF2CH2OH segment
    for a in mol.GetAtoms():
        if a.GetSymbol() != "O" or types[a.GetIdx()]:
            continue
        if a.GetFormalCharge() != 0 or a.GetDegree() != 2 or _n(a, "H") != 1 or _n(a, "C") != 1:
            raise _refuse(a, "is O outside an alcohol, sulfonate or carboxylate (ethers, esters, carbonyls and "
                             "acids with fluorinated neighbours have no published OPLS-AA parameters here)")
        c = next(x for x in a.GetNeighbors() if x.GetSymbol() == "C")
        h = next(x for x in a.GetNeighbors() if x.GetSymbol() == "H")
        if not _is_sp3_c(c) or _n(c, "F") or _n(c, "O") != 1:
            raise _refuse(a, "is an OH on a carbon that is not a plain sp3 C (gem-diols, fluorinated or aromatic "
                             "C-OH are not covered)")
        fl = [x for x in c.GetNeighbors() if x.GetSymbol() == "C" and _n(x, "F") >= 2 and _n(x, "H") == 0]
        if fl:
            if _n(c, "H") != 2 or len(fl) != 1 or _n(c, "C") != 1:
                raise _refuse(c, "is a C-OH next to a fluorinated carbon that is not -CF2CH2OH or CF3CH2OH (Silva "
                                 "et al. 2016's recipe covers 1H,1H-perfluoroalcohols only)")
            cf = fl[0]
            nf = _n(cf, "F")
            types[c.GetIdx()], types[a.GetIdx()], types[h.GetIdx()] = "opls_160", "opls_162", "opls_163"
            for x in c.GetNeighbors():
                if x.GetSymbol() == "H":
                    types[x.GetIdx()] = "opls_165"
            if nf == 3:
                types[cf.GetIdx()] = "opls_161"
            elif nf == 2:
                rest = [x for x in cf.GetNeighbors() if x.GetSymbol() == "C" and x.GetIdx() != c.GetIdx()]
                if len(rest) != 1 or not _perfluoro_c(rest[0]):
                    raise _refuse(cf, "is the CF2 of -CF2CH2OH not bonded to a perfluoroalkyl carbon")
                types[cf.GetIdx()] = SILVA_CF2
            else:
                raise _refuse(cf, "is a CF next to CH2OH (Silva et al. 2016 covers linear CF3(CF2)nCH2OH)")
            for x in cf.GetNeighbors():
                if x.GetSymbol() == "F":
                    types[x.GetIdx()] = "opls_164"
        else:
            nh = _n(c, "H")
            types[c.GetIdx()] = {3: "opls_157", 2: "opls_157", 1: "opls_158", 0: "opls_159"}[nh]
            if nh == 3:
                raise _refuse(c, "is methanol's carbon (OPLS-AA's methanol H type opls_156 is not in this table)")
            if any(x.GetIsAromatic() for x in c.GetNeighbors()):
                raise _refuse(c, "is a benzylic C-OH (not in this table)")
            types[a.GetIdx()], types[h.GetIdx()] = "opls_154", "opls_155"

    # 3. aromatic rings
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C" or not a.GetIsAromatic() or types[a.GetIdx()]:
            continue
        ring = _benzene_ring(a)
        if ring is None or a.GetDegree() != 3 or a.GetFormalCharge():
            raise _refuse(a, "is aromatic but not in a single benzene ring (fused and heteroaromatic rings are "
                             "not covered)")
        subst = [x for x in a.GetNeighbors() if x.GetIdx() not in ring]
        if len(subst) != 1:
            raise _refuse(a, "is a ring carbon with two ring bonds out of its ring")
        s = subst[0]
        sym = s.GetSymbol()
        if sym == "H":
            types[a.GetIdx()] = "opls_145"
        elif sym == "Br":
            types[a.GetIdx()] = "opls_729"
        elif sym == "C" and _is_sp3_c(s) and _n(s, "F") == 3:
            types[a.GetIdx()] = "opls_724"
        elif sym == "C" and _is_sp3_c(s) and _n(s, "F") == 0 and _n(s, "H") in (2, 3):
            types[a.GetIdx()] = "opls_145"            # OPLS-AA: the ipso C of toluene / ethylbenzene is opls_145
        elif sym == "F":
            types[a.GetIdx()] = "F?"                   # decided below, from its ring neighbours
        else:
            raise _refuse(a, f"is a ring carbon bonded to {sym} (only H, F, Br, CF3, CH3, CH2R substituents)")
    for a in mol.GetAtoms():
        if types[a.GetIdx()] != "F?":
            continue
        ring = _benzene_ring(a)
        rn = [x for x in a.GetNeighbors() if x.GetIdx() in ring]
        kinds = [("F" if any(y.GetSymbol() == "F" for y in x.GetNeighbors()) else types[x.GetIdx()]) for x in rn]
        if kinds == ["F", "F"]:
            types[a.GetIdx()] = "opls_720"
        elif sorted(kinds) == ["F", "opls_145"]:
            types[a.GetIdx()] = "opls_727"
        elif kinds == ["opls_145", "opls_145"]:
            types[a.GetIdx()] = "opls_718"
        else:
            raise _refuse(a, f"is a ring C(F) next to {' and '.join(kinds)} (no published type for that pattern)")
    for a in mol.GetAtoms():
        if types[a.GetIdx()] == "F?":
            raise _refuse(a, "ring C(F) left untyped")  # pragma: no cover

    # 4. sp3 carbons
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C" or types[a.GetIdx()]:
            continue
        if not _is_sp3_c(a):
            raise _refuse(a, "is an sp2/sp carbon outside a benzene ring or a carboxylate (alkenes, carbonyls, "
                             "nitriles: not covered)")
        cnt = _counts(a)
        if any(s not in ("C", "H", "F") for s in cnt):
            raise _refuse(a, f"is an sp3 carbon bonded to {', '.join(s for s in cnt if s not in ('C', 'H', 'F'))}")
        nh, nf, nc = cnt.get("H", 0), cnt.get("F", 0), cnt.get("C", 0)
        carbons = [x for x in a.GetNeighbors() if x.GetSymbol() == "C"]
        on_ring = [x for x in carbons if x.GetIsAromatic()]
        if nf == 0:
            if on_ring:
                if len(on_ring) > 1 or nh not in (2, 3):
                    raise _refuse(a, "is an alkyl carbon on two rings or a ring CH/C (not covered)")
                types[a.GetIdx()] = "opls_148" if nh == 3 else "opls_149"
                continue
            types[a.GetIdx()] = {3: "opls_135", 2: "opls_136", 1: "opls_137", 4: "opls_138", 0: "opls_139"}[nh]
            continue
        if nh and nf:
            if nf == 1 and all(_hydro_c(x) and not x.GetIsAromatic() for x in carbons) and nc >= 1:
                types[a.GetIdx()] = {1: "opls_957", 2: "opls_959"}.get(nc, "")
                if not types[a.GetIdx()]:
                    raise _refuse(a, "is an alkyl fluoride carbon of an unexpected shape")  # pragma: no cover
                continue
            what = "CHF2" if nf == 2 else ("CH2F/CHF next to a fluorinated or aromatic carbon" if nf == 1 else "CHF")
            raise _refuse(a, f"is a partly fluorinated carbon ({what}): no published OPLS-AA type (CHF2 groups, "
                             "e.g. omega-H telomers, HFCs and HFEs, need their own parameters)")
        # nh == 0 and nf >= 1: a perfluorinated carbon
        if nf == 1 and nc == 3 and all(_hydro_c(x) and not x.GetIsAromatic() for x in carbons):
            types[a.GetIdx()] = "opls_960"
            continue
        if on_ring:
            if nf == 3 and nc == 1:
                types[a.GetIdx()] = "opls_725"
                continue
            raise _refuse(a, "is a fluorinated carbon on a benzene ring other than CF3")
        pf_nbrs = [x for x in carbons if _perfluoro_c(x) or types[x.GetIdx()] in ("clp_C1F", "clp_CBT", "clp_CFA",
                                                                                     "opls_161", SILVA_CF2)]
        if nf in (1, 2) and not pf_nbrs:
            raise _refuse(a, f"is a CF{'2' if nf == 2 else ''} with no perfluorinated carbon neighbour (a gem-"
                             "difluoride or isolated CF inside a hydrocarbon: W&J and Pádua cover perfluoro blocks)")
        types[a.GetIdx()] = {3: "opls_961", 2: "opls_962", 1: "opls_963", 4: "opls_964"}[nf]

    # 5. H, F, Br on their hosts
    for a in mol.GetAtoms():
        if types[a.GetIdx()]:
            continue
        sym = a.GetSymbol()
        nb = a.GetNeighbors()
        if sym in ("H", "F", "Br") and len(nb) != 1:
            raise _refuse(a, "is not bonded to exactly one atom")
        host = types[nb[0].GetIdx()] if nb else ""
        if sym == "H":
            if nb[0].GetSymbol() != "C":
                raise _refuse(a, "is an H not on carbon")  # pragma: no cover (O-H handled above)
            types[a.GetIdx()] = {"opls_145": "opls_146", "opls_957": "opls_958", "opls_959": "opls_958"}.get(
                host, "opls_140")
        elif sym == "F":
            t = {"opls_718": "opls_719", "opls_720": "opls_721", "opls_727": "opls_728", "opls_725": "opls_726",
                 "opls_957": "opls_956", "opls_959": "opls_956", "opls_960": "opls_956"}.get(host)
            if t is None and host in PERFLUORO_CARBONS:
                t = "opls_965"
            if t is None:
                raise _refuse(a, f"is an F on {host or nb[0].GetSymbol()} (no F type for that host)")
            types[a.GetIdx()] = t
        elif sym == "Br":
            if host != "opls_729":
                raise _refuse(a, "is a Br not on a benzene ring (alkyl bromides are not in this table)")
            types[a.GetIdx()] = "opls_730"
        else:
            raise _refuse(a, "left untyped")  # pragma: no cover

    # 6. charges
    charges = [0.0] * n
    charge_notes: dict[int, str] = {}
    for i, t in enumerate(types):
        q = TYPES[t].charge
        if q is not None:
            charges[i] = q
    for i, t in enumerate(types):
        if t == SILVA_CF2:
            a = mol.GetAtomWithIdx(i)
            ch2 = next(x for x in a.GetNeighbors() if types[x.GetIdx()] == "opls_160")
            seg = [x.GetIdx() for x in a.GetNeighbors() if x.GetSymbol() == "F"]
            seg += [ch2.GetIdx()] + [x.GetIdx() for x in ch2.GetNeighbors() if x.GetIdx() != i]
            o = next(x for x in ch2.GetNeighbors() if x.GetSymbol() == "O")
            seg += [x.GetIdx() for x in o.GetNeighbors() if x.GetSymbol() == "H"]
            charges[i] = round(-sum(charges[j] for j in seg), 6)
            charge_notes[i] = f"{charges[i]:+.3f} e, {SILVA_RULE}"
    total = sum(charges)
    formal = Chem.GetFormalCharge(mol)
    if abs(total - formal) > 1e-6:
        raise PFTypingError(f"the published charges of these types sum to {total:+.4f} e, not the formal charge "
                            f"{formal:+d}: the groups do not combine into a neutral (or -1) molecule without "
                            "redistributing charge, which this table does not do")
    fam = {TYPES[t].family for t in types}
    junction = any(
        ((types[b.GetBeginAtomIdx()] in PERFLUORO_CARBONS | {"opls_161", SILVA_CF2}) and
         _hydro_c(b.GetEndAtom()) and types[b.GetEndAtomIdx()] not in ("opls_160",)) or
        ((types[b.GetEndAtomIdx()] in PERFLUORO_CARBONS | {"opls_161", SILVA_CF2}) and
         _hydro_c(b.GetBeginAtom()) and types[b.GetBeginAtomIdx()] not in ("opls_160",))
        for b in mol.GetBonds())
    if any(t in PERFLUORO_CARBONS for t in types) and mol.GetRingInfo().NumRings() and "aromatic" not in fam:
        notes.append("a ring of perfluoro carbons: Watkins & Jorgensen fitted linear chains (their Table 6 also "
                     "tests c-C4F8 and c-C5F10)")
    return PFAssignment(types, charges, charge_notes, fam, junction, notes)


# ---------------------------------------------------------------------------------------- the recipes
#: published recipes: (families the molecule's types may come from, needs a junction?, label, reference keys)
RECIPES: list[tuple[frozenset[str], bool | None, str, tuple[str, ...]]] = [
    (frozenset({"alkane"}), False, "OPLS-AA alkanes (Jorgensen et al. 1996)", ("jorgensen1996",)),
    (frozenset({"perfluoroalkane"}), False, "OPLS-AA perfluoroalkanes (Watkins & Jorgensen 2001; liquid density "
     "and ΔHvap in their Table 6, σ in Tsige & Grest 2008)", ("watkins2001",)),
    (frozenset({"aromatic"}), False, "OPLS-AA benzene and its F/CF3/CH3/Br derivatives (GROMACS oplsaa.ff; the "
     "fluorinated ring types were added for Caleman et al. 2012's liquid benchmark)", ("jorgensen1996", "caleman2012")),
    (frozenset({"aromatic", "alkane"}), False, "OPLS-AA alkylbenzenes (GROMACS oplsaa.ff)", ("jorgensen1996",)),
    (frozenset({"alcohol", "alkane"}), False, "OPLS-AA alcohols (Jorgensen et al. 1996)", ("jorgensen1996",)),
    (frozenset({"alkyl_fluoride", "alkane"}), False, "OPLS-AA monoalkyl fluorides (opls_956-960; GROMACS marks "
     "them tentative)", ("jorgensen1996",)),
    (frozenset({"perfluoroalkane", "alkane"}), True, "semifluorinated alkane: W&J 2001 + OPLS-AA with Pádua 2002's "
     "junction torsions (as Pierce et al. 2008 and the Lisbon group; Pierce's 25 % weaker H-F Lennard-Jones cross "
     "term is NOT applied: global geometric mixing)", ("watkins2001", "padua2002")),
    (frozenset({"tfe"}), False, "OPLS-AA 2,2,2-trifluoroethanol (opls_160-165)", ("silva2016",)),
    (frozenset({"tfe", "perfluoroalkane"}), None, "1H,1H-perfluoroalcohol: Silva et al. 2016 (W&J tail, TFE "
     "parameters for -CF2CH2OH, Pádua's junction torsions; liquid density within 2 %)", ("silva2016",)),
    (frozenset({"sulfonate"}), False, "CL&P triflate anion (Canongia Lopes & Pádua 2004)", ("clp2004",)),
    (frozenset({"sulfonate", "perfluoroalkane"}), False, "perfluoroalkanesulfonate anion: W&J chain + CL&P's "
     "perfluoroalkanesulfonyl head (the combination Damião et al. 2023 describe; their own files not checked)",
     ("watkins2001", "shimizu2010", "clp2004")),
    (frozenset({"carboxylate"}), False, "CL&P trifluoroacetate anion", ("clp_tfa",)),
    (frozenset({"ion"}), False, "OPLS-AA monatomic cation (Åqvist 1990)", ("aqvist1990",)),
]


def recipe_of(asg: PFAssignment) -> tuple[str, str]:
    """(tier, recipe): tier "published" when the molecule follows a published recipe, "composed" when it combines
    published groups no paper combined (each term still from a source)."""
    for fams, junction, label, _refs in RECIPES:
        if asg.families == set(fams) and (junction is None or junction == asg.junction):
            return "published", label
    return "composed", ("combines " + ", ".join(sorted(asg.families)) + " groups (every type and term from a "
                        "published table, the combination not validated as such)")


# ------------------------------------------------------------------------------------------- bonded terms
@dataclass(frozen=True)
class Term:
    """One bonded type in LAMMPS ``real`` units: style, coefficients, label and source."""

    style: str
    coeffs: tuple[float, ...]
    label: str
    source: str
    transfer: str = ""          # non-empty: the term is a labelled transfer (what was read as what, and why)


def _gmx_bt(t: str) -> str:
    return TYPES[t].btype


def _clp(t: str) -> str:
    return TYPES[t].clp


def _lookup(table: dict, key: tuple[str, ...], wildcards: bool = False):
    for cand in (key, key[::-1]):
        if cand in table:
            return cand
    if wildcards and len(key) == 4:
        for cand in ((key[0], key[1], key[2], "X"), ("X", key[1], key[2], key[3]), (key[3], key[2], key[1], "X"),
                     ("X", key[2], key[1], key[0]), ("X", key[1], key[2], "X"), ("X", key[2], key[1], "X")):
            if cand in table:
                return cand
    return None


def _touches_clp(types: tuple[str, ...]) -> bool:
    return any(t.startswith("clp_") for t in types)


def _junction_term(types: tuple[str, ...]) -> bool:
    fl = any(_clp(t) == "CF" for t in types if TYPES[t].family in ("perfluoroalkane", "tfe"))
    hy = any(_clp(t) == "CT" for t in types if TYPES[t].element == "C")
    return fl and hy


def bond_parameters(a: str, b: str) -> Term:
    if _touches_clp((a, b)):
        k = _lookup(CLP_BONDS, (_clp(a), _clp(b)))
        if k:
            r0, kk, line = CLP_BONDS[k]
            return Term("harmonic", (kk / 2.0 / KJ_PER_KCAL, r0), "clp " + "-".join(k), f"CL&P il.ff line {line}")
    k = _lookup(GMX_BONDS, (_gmx_bt(a), _gmx_bt(b)))
    if k:
        b0, kb, line, c = GMX_BONDS[k]
        return Term("harmonic", (kb / 2.0 / KJ_PER_KCAL / 100.0, b0 * 10.0), "-".join(k),
                    f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({c})")
    raise PFTypingError(f"no published bond parameters for {a}-{b}")


def angle_parameters(a: str, b: str, c: str) -> Term:
    if _touches_clp((a, b, c)):
        k = _lookup(CLP_ANGLES, (_clp(a), _clp(b), _clp(c)))
        if k:
            th, kk, line = CLP_ANGLES[k]
            return Term("harmonic", (kk / 2.0 / KJ_PER_KCAL, th), "clp " + "-".join(k), f"CL&P il.ff line {line}")
    k = _lookup(GMX_ANGLES, (_gmx_bt(a), _gmx_bt(b), _gmx_bt(c)))
    if k:
        th, kk, line, cm = GMX_ANGLES[k]
        return Term("harmonic", (kk / 2.0 / KJ_PER_KCAL, th), "-".join(k),
                    f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({cm})")
    if _junction_term((a, b, c)):
        k = _lookup(CLP_ANGLES, (_clp(a), _clp(b), _clp(c)))
        if k:
            th, kk, line = CLP_ANGLES[k]
            return Term("harmonic", (kk / 2.0 / KJ_PER_KCAL, th), "clp " + "-".join(k),
                        f"CL&P il.ff line {line} (fluoroalkyl junction, W&J 2001 / Pádua 2002)")
    raise PFTypingError(f"no published angle parameters for {a}-{b}-{c}")


def _opls(v_kj: tuple[float, ...]) -> tuple[float, float, float, float]:
    return tuple(x / KJ_PER_KCAL + 0.0 for x in v_kj)   # type: ignore[return-value]


def dihedral_parameters(a: str, b: str, c: str, d: str) -> Term:
    """The proper dihedral a-b-c-d (type names), looked up in this order:

    1. a term touching a CL&P anion head: CL&P by class;
    2. four W&J perfluoroalkyl carbons: Watkins & Jorgensen's C-C-C-C torsion (as ``md.oplsaa``);
    3. a GROMACS term that names the TFE bond type CT_4 (Duffy's trifluoroethanol terms);
    4. a fluoroalkyl / hydrocarbon junction: Pádua 2002's terms from CL&P, by class;
    5. GROMACS by bond type (exact, then X wildcards), as GROMACS itself looks it up.
    """
    tt = (a, b, c, d)
    if _touches_clp(tt):
        k = _lookup(CLP_DIHEDRALS, tuple(_clp(t) for t in tt))
        if k:
            v, line = CLP_DIHEDRALS[k]
            return Term("opls", _opls(v), "clp " + "-".join(k), f"CL&P il.ff line {line}")
    if set(tt) <= PERFLUORO_CARBONS | {SILVA_CF2}:
        label, v, src = O.dihedral_parameters(*(("opls_962" if t == SILVA_CF2 else t) for t in tt))
        return Term("opls", tuple(v), label, src)
    bts = tuple(_gmx_bt(t) for t in tt)
    if "CT_4" in bts:
        k = _lookup(GMX_DIHEDRALS, bts)
        if k:
            rb, line, cm = GMX_DIHEDRALS[k]
            return Term("opls", O.rb_to_opls(rb), "-".join(k), f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({cm})")
    if _junction_term(tt):
        k = _lookup(CLP_DIHEDRALS, tuple(_clp(t) for t in tt))
        if k and line_is_junction(CLP_DIHEDRALS[k][1]):
            v, line = CLP_DIHEDRALS[k]
            return Term("opls", _opls(v), "padua " + "-".join(k),
                        f"CL&P il.ff line {line} (fluoroalkyl/hydrocarbon junction, Pádua 2002)")
    k = _lookup(GMX_DIHEDRALS, bts, wildcards=True)
    if k:
        rb, line, cm = GMX_DIHEDRALS[k]
        return Term("opls", O.rb_to_opls(rb), "-".join(k), f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({cm})")
    if "CT_4" in bts:
        plain = tuple("CT" if x == "CT_4" else x for x in bts)
        k = _lookup(GMX_DIHEDRALS, plain, wildcards=True)
        if k:
            rb, line, cm = GMX_DIHEDRALS[k]
            return Term("opls", O.rb_to_opls(rb), "-".join(k) + " (CT_4 as CT)",
                        f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({cm})",
                        transfer=f"{'-'.join(bts)} has no term in GROMACS, CL&P or Pádua 2002: the TFE CH2 (bond type "
                                 f"CT_4) read as a plain CT, i.e. the hydrocarbon {'-'.join(k)} term")
    raise PFTypingError(f"no published torsion parameters for {'-'.join(tt)} (bond types {'-'.join(bts)})")


def line_is_junction(line: int) -> bool:
    """CL&P il.ff lines 455-461 hold the fluoroalkyl / hydrocarbon junction torsions (Pádua 2002)."""
    return 455 <= line <= 461


def improper_aromatic() -> Term:
    k, d, n = O.lammps_improper_ca()
    return Term("cvff", (k, d, n), "Z-CA-X-Y",
                f"GROMACS oplsaa.ff/ffbonded.itp line {O.IMPROPER_CA_LINE} (improper_Z_CA_X_Y); not in foyer's XML")


def improper_carboxylate() -> Term:
    """CL&P's carbonyl improper (V2 = 87.864 kJ/mol, ½V2(1 - cos 2φ)) as cvff K [1 + d cos 2φ], K = V2/2, d = -1."""
    v, line = CLP_IMPROPERS[("CF", "O2", "CO", "O2")]
    return Term("cvff", (v[1] / 2.0 / KJ_PER_KCAL, -1, 2), "clp CF-O2-CO-O2", f"CL&P il.ff line {line}")
