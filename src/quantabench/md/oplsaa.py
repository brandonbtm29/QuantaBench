"""OPLS-AA atom types and parameters for the liquid-slab tools (MD-2): alkanes, perfluoroalkanes, benzene and
fluorobenzenes.

Scope, and why it is small
--------------------------
QuantaBench types a molecule for OPLS-AA with the rules in ``assign_types`` (RDKit, no SMARTS engine) and takes
the parameters from the table below. The table holds only the atom types these rules can give; every other
molecule is refused with a message that names the cross-check force fields (``md.forcefield``: OpenFF Sage,
GAFF2), never typed by guess. Supported:

* alkanes, linear, branched and cyclic (opls_135-140);
* perfluoroalkanes, linear, branched and cyclic, CF4 included (Watkins & Jorgensen 2001: opls_961-965);
* benzene and its (per)fluorinated derivatives (opls_145/146, opls_718/719).

Semifluorinated chains (a CF2 bonded to a CH2) are refused: their junction needs extra H-F terms (the modified
OPLS-AA of Pierce et al., J. Chem. Phys. 2008, 128, 214903) that are not in the table.

Where every number comes from
-----------------------------
Each entry stores the values exactly as they are written in GROMACS' distribution of OPLS-AA
(``share/top/oplsaa.ff``, LGPL-2.1, commit ``GROMACS_COMMIT``) with the line number, in its units (nm, kJ/mol,
degrees); conversion to LAMMPS ``real`` units happens in one place (``lammps_*`` functions). The same values are in
foyer's ``oplsaa.xml`` (MIT, commit ``FOYER_COMMIT``), which is cross-checked line by line by
``scripts/check_oplsaa_params.py`` (run it with the two files to re-verify after an edit).

Two deliberate differences from foyer's XML, both following GROMACS:

1. **The perfluoroalkane C-C-C-C torsion.** Watkins & Jorgensen fitted their own C-C-C-C torsion for
   perfluoroalkanes (it gives the helical twist of the chain). The perfluoro carbons have the bond type ``CT``
   of the alkanes, so a lookup by bond type finds the hydrocarbon torsion; GROMACS therefore carries the
   perfluoro torsion as a ``#define`` that a topology must apply explicitly (ffbonded.itp,
   ``dih_perfluoroalkane_CT_CT_CT_CT``), and foyer, which looks up by class only, does not apply it. QuantaBench
   applies it to every C-C-C-C dihedral whose four carbons are perfluoroalkyl carbons (opls_961-964).
2. **Aromatic impropers.** foyer's XML has no improper torsions; GROMACS' OPLS-AA keeps aromatic carbons planar
   with ``improper_Z_CA_X_Y`` (180°, 4.6024 kJ/mol, n = 2), applied here to every aromatic carbon.

Nonbonded conventions of OPLS-AA: geometric combining rules for σ and ε, 1-4 interactions scaled by 0.5 for both
Lennard-Jones and Coulomb (GROMACS forcefield.itp: ``comb-rule 3``, ``fudgeLJ 0.5``, ``fudgeQQ 0.5``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem

KJ_PER_KCAL = 4.184

GROMACS_COMMIT = "bc88adaf4e6c918646aba6d3495d3383c6aca6b4"
GROMACS_URL = f"https://github.com/gromacs/gromacs/tree/{GROMACS_COMMIT}/share/top/oplsaa.ff"
FOYER_COMMIT = "89b7fa867d1ce059fcb474123cf84ccc373d0248"
FOYER_URL = f"https://github.com/mosdef-hub/foyer/blob/{FOYER_COMMIT}/foyer/forcefields/xml/oplsaa.xml"
FOYER_SHA256 = "c78ccb763cda33e3456a3f8b4ed8a8f0361b10c92c33aa02f2a9abf618c163e4"

REFERENCES = {
    "jorgensen1996": "Jorgensen, Maxwell, Tirado-Rives. Development and testing of the OPLS all-atom force field on "
                     "conformational energetics and properties of organic liquids. J. Am. Chem. Soc. 1996, 118, "
                     "11225-11236. doi:10.1021/ja9621760",
    "watkins2001": "Watkins, Jorgensen. Perfluoroalkanes: conformational analysis and liquid-state properties from "
                   "ab initio and Monte Carlo calculations. J. Phys. Chem. A 2001, 105, 4118-4125. doi:10.1021/jp004071w",
    "jorgensen2004": "Jorgensen, Ulmschneider, Tirado-Rives. Free energies of hydration from a generalized Born model "
                     "and an all-atom force field. J. Phys. Chem. B 2004, 108, 16264-16270. doi:10.1021/jp0484579 "
                     "(the aromatic C-F types, as cited by foyer's oplsaa.xml)",
    "gromacs": f"GROMACS share/top/oplsaa.ff (ffnonbonded.itp, ffbonded.itp), LGPL-2.1, commit {GROMACS_COMMIT[:10]}",
    "foyer": f"foyer foyer/forcefields/xml/oplsaa.xml, MIT, commit {FOYER_COMMIT[:10]} (cross-check)",
}

#: GROMACS forcefield.itp [ defaults ]: comb-rule 3 (geometric σ and ε), fudgeLJ 0.5, fudgeQQ 0.5
MIXING = "geometric"
SPECIAL_LJ = (0.0, 0.0, 0.5)
SPECIAL_COUL = (0.0, 0.0, 0.5)


class OplsTypingError(ValueError):
    """The molecule has an atom the OPLS-AA rules here do not cover."""


@dataclass(frozen=True)
class AtomType:
    """One row of GROMACS' ffnonbonded.itp: name, bond type, mass, charge, σ (nm), ε (kJ/mol)."""

    name: str
    bond_type: str
    element: str
    mass: float
    charge: float
    sigma_nm: float
    epsilon_kj: float
    description: str
    reference: str
    line: int                     # line in ffnonbonded.itp at GROMACS_COMMIT

    @property
    def sigma_A(self) -> float:
        return self.sigma_nm * 10.0

    @property
    def epsilon_kcal(self) -> float:
        return self.epsilon_kj / KJ_PER_KCAL

    @property
    def source(self) -> str:
        return (f"OPLS-AA {self.name} ({self.description}): GROMACS oplsaa.ff/ffnonbonded.itp line {self.line} "
                f"(= foyer oplsaa.xml); {REFERENCES[self.reference].split('.')[0]} et al. "
                f"[{self.reference}]")


def _t(name, bt, el, mass, q, sig, eps, desc, ref, line) -> AtomType:
    return AtomType(name, bt, el, mass, q, sig, eps, desc, ref, line)


ATOM_TYPES: dict[str, AtomType] = {t.name: t for t in (
    _t("opls_135", "CT", "C", 12.01100, -0.180, 0.350, 0.276144, "alkane CH3", "jorgensen1996", 145),
    _t("opls_136", "CT", "C", 12.01100, -0.120, 0.350, 0.276144, "alkane CH2", "jorgensen1996", 146),
    _t("opls_137", "CT", "C", 12.01100, -0.060, 0.350, 0.276144, "alkane CH", "jorgensen1996", 147),
    _t("opls_138", "CT", "C", 12.01100, -0.240, 0.350, 0.276144, "alkane CH4", "jorgensen1996", 148),
    _t("opls_139", "CT", "C", 12.01100, 0.000, 0.350, 0.276144, "alkane C", "jorgensen1996", 149),
    _t("opls_140", "HC", "H", 1.00800, 0.060, 0.250, 0.125520, "alkane H", "jorgensen1996", 150),
    _t("opls_145", "CA", "C", 12.01100, -0.115, 0.355, 0.292880, "benzene C", "jorgensen1996", 155),
    _t("opls_146", "HA", "H", 1.00800, 0.115, 0.242, 0.125520, "benzene H", "jorgensen1996", 157),
    _t("opls_718", "CA", "C", 12.01100, 0.220, 0.355, 0.292880, "fluorobenzene C(F)", "jorgensen2004", 700),
    _t("opls_719", "F", "F", 18.99840, -0.220, 0.285, 0.255224, "fluorobenzene F", "jorgensen2004", 701),
    _t("opls_961", "CT", "C", 12.01100, 0.360, 0.350, 0.276144, "perfluoroalkane CF3", "watkins2001", 815),
    _t("opls_962", "CT", "C", 12.01100, 0.240, 0.350, 0.276144, "perfluoroalkane CF2", "watkins2001", 816),
    _t("opls_963", "CT", "C", 12.01100, 0.120, 0.350, 0.276144, "perfluoroalkane CF", "watkins2001", 817),
    _t("opls_964", "CT", "C", 12.01100, 0.480, 0.350, 0.405848, "perfluoroalkane CF4", "watkins2001", 818),
    _t("opls_965", "F", "F", 18.99840, -0.120, 0.295, 0.221752, "perfluoroalkane F", "watkins2001", 819),
)}

PERFLUORO_CARBONS = frozenset({"opls_961", "opls_962", "opls_963", "opls_964"})

#: [ bondtypes ] funct 1: (b0 nm, kb kJ mol⁻¹ nm⁻²), V = ½ kb (b - b0)²; key sorted
BONDS: dict[tuple[str, str], tuple[float, float, int, str]] = {
    ("CA", "CA"): (0.14000, 392459.2, 97, "TRP,TYR,PHE"),
    ("CA", "HA"): (0.10800, 307105.6, 140, "PHE, etc."),
    ("CT", "CT"): (0.15290, 224262.4, 192, "CHARMM 22 parameter file"),
    ("CT", "HC"): (0.10900, 284512.0, 196, "CHARMM 22 parameter file"),
    ("CT", "F"): (0.13320, 307105.6, 276, "PAK CT-F for CHF3"),
    ("CA", "F"): (0.13540, 351456.0, 277, "wlj"),
}

#: [ angletypes ] funct 1: (θ0 degrees, k kJ mol⁻¹ rad⁻²), V = ½ k (θ - θ0)²; key with the ends sorted
ANGLES: dict[tuple[str, str, str], tuple[float, float, int, str]] = {
    ("CA", "CA", "F"): (120.000, 669.440, 530, "wlj"),
    ("CA", "CA", "CA"): (120.000, 527.184, 587, "PHE(OL)"),
    ("CA", "CA", "HA"): (120.000, 292.880, 603, ""),
    ("CT", "CT", "CT"): (112.700, 488.273, 873, "CHARMM 22 parameter file"),
    ("CT", "CT", "HC"): (110.700, 313.800, 881, "CHARMM 22 parameter file"),
    ("HC", "CT", "HC"): (107.800, 276.144, 929, "CHARMM 22 parameter file"),
    ("CT", "CT", "F"): (109.500, 418.400, 1187, "PAK (emd 5-09-94)"),
    ("F", "CT", "F"): (109.100, 644.336, 1190, "PAK F-CT-F (emd 5-09-94)"),
}

#: [ dihedraltypes ] funct 3, Ryckaert-Bellemans C0..C5 (kJ/mol), V = Σ Cn cosⁿ(φ - 180°); "X" = any
DIHEDRALS_RB: dict[tuple[str, str, str, str], tuple[tuple[float, ...], int, str]] = {
    ("CT", "CT", "CT", "F"): ((1.46440, 1.88280, 0.0, -3.34720, 0.0, 0.0), 1593, "alkyl fluoride"),
    ("CT", "CT", "CT", "CT"): ((2.92880, -1.46440, 0.20920, -1.67360, 0.0, 0.0), 1596, "hydrocarbon all-atom"),
    ("CT", "CT", "CT", "HC"): ((0.62760, 1.88280, 0.0, -2.51040, 0.0, 0.0), 1599, "hydrocarbon all-atom"),
    ("F", "CT", "CT", "F"): ((-5.23000, 5.23000, 0.0, 0.0, 0.0, 0.0), 1756, "1,2-difluoride"),
    ("HC", "CT", "CT", "HC"): ((0.62760, 1.88280, 0.0, -2.51040, 0.0, 0.0), 1824, "hydrocarbon *new* 11/99"),
    ("X", "CA", "CA", "X"): ((30.33400, 0.0, -30.33400, 0.0, 0.0, 0.0), 1956, "aromatic ring"),
}

#: ffbonded.itp line 2485: "#define dih_perfluoroalkane_CT_CT_CT_CT" (RB, kJ/mol), Watkins & Jorgensen 2001
PERFLUORO_CCCC_RB = (14.91596, -22.56431, -39.41328, 11.61479, 35.44685, 0.0)
PERFLUORO_CCCC_LINE = 2485

#: ffbonded.itp line 2571: "#define improper_Z_CA_X_Y 180.0 4.60240 2" (periodic: V = k (1 + cos(nφ - φs)))
IMPROPER_CA = (180.0, 4.60240, 2)
IMPROPER_CA_LINE = 2571


# ----------------------------------------------------------------------------------------------- conversions
def rb_to_opls(c: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Ryckaert-Bellemans C0..C5 (kJ/mol, GROMACS) -> OPLS Fourier V1..V4 (kcal/mol, LAMMPS ``dihedral_style
    opls``: E = ½V1(1+cos φ) + ½V2(1-cos 2φ) + ½V3(1+cos 3φ) + ½V4(1-cos 4φ)).

    From the GROMACS manual's relations C0 = V2 + (V1+V3)/2, C1 = (-V1+3V3)/2, C2 = -V2+4V4, C3 = -2V3, C4 = -4V4,
    C5 = 0. A set that the four-term form cannot represent (C5 ≠ 0, or a C0 that is not the constant above) is
    refused rather than silently changed."""
    c0, c1, c2, c3, c4, c5 = (list(c) + [0.0] * 6)[:6]
    if abs(c5) > 1e-9:
        raise ValueError(f"RB term C5 = {c5} has no OPLS Fourier equivalent")
    v4 = -c4 / 4.0
    v3 = -c3 / 2.0
    v2 = -c2 - c4
    v1 = -2.0 * c1 - 1.5 * c3
    if abs(c0 - (v2 + 0.5 * (v1 + v3))) > 1e-4:
        raise ValueError(f"RB C0 = {c0} is not the Fourier constant {v2 + 0.5 * (v1 + v3)}")
    return tuple(v / KJ_PER_KCAL + 0.0 for v in (v1, v2, v3, v4))   # + 0.0: no -0.0 in the files


def lammps_bond(key: tuple[str, str]) -> tuple[float, float]:
    """(K kcal/mol/Å², r0 Å) for ``bond_style harmonic`` (E = K (r - r0)², so K = kb/2)."""
    b0, kb, _line, _c = BONDS[key]
    return kb / 2.0 / KJ_PER_KCAL / 100.0, b0 * 10.0


def lammps_angle(key: tuple[str, str, str]) -> tuple[float, float]:
    """(K kcal/mol/rad², θ0 degrees) for ``angle_style harmonic`` (E = K (θ - θ0)², so K = k/2)."""
    theta, k, _line, _c = ANGLES[key]
    return k / 2.0 / KJ_PER_KCAL, theta


def lammps_improper_ca() -> tuple[float, int, int]:
    """(K kcal/mol, d, n) for ``improper_style cvff``: E = K [1 + d cos(nφ)], d = cos φs = -1 for φs = 180°."""
    phase, k, n = IMPROPER_CA
    return k / KJ_PER_KCAL, int(round(math.cos(math.radians(phase)))), int(n)


# --------------------------------------------------------------------------------------------------- typing
def _benzene_ring(atom: Chem.Atom) -> bool:
    """The atom is in a six-membered aromatic ring of carbons only."""
    ri = atom.GetOwningMol().GetRingInfo()
    for ring in ri.AtomRings():
        if atom.GetIdx() in ring and len(ring) == 6:
            mol = atom.GetOwningMol()
            if all(mol.GetAtomWithIdx(i).GetSymbol() == "C" and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                return True
    return False


def _refuse(atom: Chem.Atom, why: str) -> OplsTypingError:
    return OplsTypingError(
        f"OPLS-AA typing here covers alkanes, perfluoroalkanes and (fluoro)benzenes only: atom {atom.GetIdx()} "
        f"({atom.GetSymbol()}) {why}. Try --ff oplsaa-pf (more fluorinated groups, from published tables), or "
        "--ff openff / --ff gaff2 (cross-checks) for this molecule.")


def _carbon_type(atom: Chem.Atom) -> str:
    if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons():
        raise _refuse(atom, "is charged or a radical")
    nbrs = [n.GetSymbol() for n in atom.GetNeighbors()]
    if atom.GetIsAromatic():
        if not _benzene_ring(atom) or atom.GetDegree() != 3:
            raise _refuse(atom, "is aromatic but not in a benzene ring")
        ring_nbrs = [n for n in atom.GetNeighbors() if n.GetIsAromatic()]
        subst = [n.GetSymbol() for n in atom.GetNeighbors() if not n.GetIsAromatic()]
        if len(ring_nbrs) != 2 or len(subst) != 1:
            raise _refuse(atom, "is a fused or bridging aromatic carbon")
        if subst[0] == "H":
            return "opls_145"
        if subst[0] == "F":
            return "opls_718"
        raise _refuse(atom, f"is an aromatic carbon bonded to {subst[0]}")
    if atom.GetHybridization() != Chem.HybridizationType.SP3 or atom.GetDegree() != 4:
        raise _refuse(atom, "is not an sp3 or benzene carbon")
    if any(s not in ("C", "H", "F") for s in nbrs):
        raise _refuse(atom, "has a neighbor other than C, H or F")
    n_h, n_f = nbrs.count("H"), nbrs.count("F")
    carbons = [n for n in atom.GetNeighbors() if n.GetSymbol() == "C"]
    if any(c.GetIsAromatic() for c in carbons):
        raise _refuse(atom, "is an alkyl carbon on an aromatic ring (toluene-like types are not in the table)")
    if n_h and n_f:
        raise _refuse(atom, "carries both H and F (partly fluorinated carbons are not in the table)")
    if n_f == 0:
        if any(any(x.GetSymbol() == "F" for x in c.GetNeighbors()) for c in carbons):
            raise _refuse(atom, "joins a hydrocarbon and a fluorocarbon block (semifluorinated junctions need "
                                "the extra H-F terms of Pierce et al. 2008)")
        return {3: "opls_135", 2: "opls_136", 1: "opls_137", 4: "opls_138", 0: "opls_139"}[n_h]
    if any(any(x.GetSymbol() == "H" for x in c.GetNeighbors()) for c in carbons):
        raise _refuse(atom, "joins a fluorocarbon and a hydrocarbon block (semifluorinated junctions need the "
                            "extra H-F terms of Pierce et al. 2008)")
    return {3: "opls_961", 2: "opls_962", 1: "opls_963", 4: "opls_964"}[n_f]


def assign_types(mol: Chem.Mol) -> list[str]:
    """OPLS-AA type name of every atom of ``mol`` (explicit hydrogens required), or ``OplsTypingError``."""
    if any(a.GetTotalNumHs(includeNeighbors=False) > 0 for a in mol.GetAtoms() if a.GetSymbol() != "H"):
        raise OplsTypingError("the molecule needs explicit hydrogens (Chem.AddHs)")
    types: list[str] = [""] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            types[atom.GetIdx()] = _carbon_type(atom)
        elif atom.GetSymbol() not in ("H", "F"):
            raise _refuse(atom, "is not C, H or F")
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ("H", "F"):
            continue
        nb = atom.GetNeighbors()
        if len(nb) != 1 or nb[0].GetSymbol() != "C":
            raise _refuse(atom, "is not bonded to exactly one carbon")
        host = types[nb[0].GetIdx()]
        if sym == "H":
            types[atom.GetIdx()] = "opls_146" if host == "opls_145" else "opls_140"
        else:
            types[atom.GetIdx()] = "opls_719" if host == "opls_718" else "opls_965"
    return types


def bond_key(a: str, b: str) -> tuple[str, str]:
    k = (ATOM_TYPES[a].bond_type, ATOM_TYPES[b].bond_type)
    for cand in (k, k[::-1]):
        if cand in BONDS:
            return cand
    raise OplsTypingError(f"no OPLS-AA bond parameters for {k[0]}-{k[1]}")


def angle_key(a: str, b: str, c: str) -> tuple[str, str, str]:
    k = (ATOM_TYPES[a].bond_type, ATOM_TYPES[b].bond_type, ATOM_TYPES[c].bond_type)
    for cand in (k, k[::-1]):
        if cand in ANGLES:
            return cand
    raise OplsTypingError(f"no OPLS-AA angle parameters for {'-'.join(k)}")


def dihedral_parameters(a: str, b: str, c: str, d: str) -> tuple[str, tuple[float, float, float, float], str]:
    """(label, V1..V4 kcal/mol, source) of the proper dihedral a-b-c-d (type names). Perfluoroalkyl C-C-C-C
    takes Watkins & Jorgensen's torsion; otherwise the bond-type lookup GROMACS makes (exact, then X-b-c-X)."""
    if {a, b, c, d} <= PERFLUORO_CARBONS:
        return ("perfluoro CT-CT-CT-CT", rb_to_opls(PERFLUORO_CCCC_RB),
                f"GROMACS oplsaa.ff/ffbonded.itp line {PERFLUORO_CCCC_LINE} (dih_perfluoroalkane_CT_CT_CT_CT); "
                "Watkins & Jorgensen 2001 [watkins2001]")
    k = tuple(ATOM_TYPES[t].bond_type for t in (a, b, c, d))
    for cand in (k, k[::-1], ("X", k[1], k[2], "X"), ("X", k[2], k[1], "X")):
        if cand in DIHEDRALS_RB:
            rb, line, comment = DIHEDRALS_RB[cand]
            return ("-".join(cand), rb_to_opls(rb),
                    f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({comment}) = foyer oplsaa.xml <Proper>")
    raise OplsTypingError(f"no OPLS-AA torsion parameters for {'-'.join(k)}")
