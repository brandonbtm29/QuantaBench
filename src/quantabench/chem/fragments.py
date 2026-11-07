"""Fragment library of the 3D model kit (UI-7): groups, rings, substituents and PFAS building blocks, as data.

A fragment is a SMILES string with a dummy atom ``[*:1]`` where it bonds to the molecule being built; the
dummy's neighbour is the *attachment atom*. A second dummy ``[*:2]`` marks where a chain continues: the kit
turns it into a hydrogen, the free valence the next fragment attaches to (``(CF2)n``, the ether O, the
fluorotelomer linker). The 3D geometry is made when a fragment is first used (``chem.edit``), so this module is
plain data plus a few builders.

The 2D sketcher (UI-6) keeps its own short template list for now; both will read this library later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem, rdBase

#: Palette tabs, in the order the model kit shows them.
CATEGORIES = ("group", "ring", "substituent", "pfas")


@dataclass(frozen=True)
class Fragment:
    name: str          # unique, lower case: "carboxylic acid"
    smiles: str        # with [*:1] at the attachment point (and [*:2] where a chain continues)
    category: str      # one of CATEGORIES
    label: str = ""    # short palette caption: "COOH"
    note: str = ""

    @property
    def caption(self) -> str:
        return self.label or self.name


_LIBRARY: tuple[Fragment, ...] = (
    # ---- groups (Spartan's Groups list)
    Fragment("methyl", "[*:1]C", "group", "CH3"),
    Fragment("ethyl", "[*:1]CC", "group", "C2H5"),
    Fragment("hydroxyl", "[*:1]O", "group", "OH"),
    Fragment("amine", "[*:1]N", "group", "NH2"),
    Fragment("carbonyl", "[*:1]C(=O)[*:2]", "group", "C=O", "a -C(=O)- link; on its own a formyl group"),
    Fragment("carboxylic acid", "[*:1]C(=O)O", "group", "COOH"),
    Fragment("carboxylate", "[*:1]C(=O)[O-]", "group", "COO-"),
    Fragment("ester", "[*:1]C(=O)O[*:2]", "group", "COO", "a -C(=O)O- link: attach the alkyl group at the O"),
    Fragment("amide", "[*:1]C(=O)N[*:2]", "group", "CONH", "a -C(=O)NH- link"),
    Fragment("nitro", "[*:1][N+](=O)[O-]", "group", "NO2"),
    Fragment("nitrile", "[*:1]C#N", "group", "CN"),
    Fragment("thiol", "[*:1]S", "group", "SH"),
    Fragment("sulfonic acid", "[*:1]S(=O)(=O)O", "group", "SO3H"),
    Fragment("sulfonate", "[*:1]S(=O)(=O)[O-]", "group", "SO3-"),
    Fragment("sulfonamide", "[*:1]S(=O)(=O)N", "group", "SO2NH2"),
    Fragment("phosphate", "[*:1]OP(=O)(O)O", "group", "OPO3H2", "phosphate monoester (PAPs)"),
    Fragment("phosphonic acid", "[*:1]P(=O)(O)O", "group", "PO3H2", "PFPA head group"),
    # ---- rings
    Fragment("phenyl", "[*:1]c1ccccc1", "ring", "Ph"),
    Fragment("cyclohexyl", "[*:1]C1CCCCC1", "ring", "C6H11", "chair"),
    Fragment("2-pyridyl", "[*:1]c1ccccn1", "ring", "2-Py"),
    Fragment("3-pyridyl", "[*:1]c1cccnc1", "ring", "3-Py"),
    Fragment("4-pyridyl", "[*:1]c1ccncc1", "ring", "4-Py"),
    Fragment("cyclopentyl", "[*:1]C1CCCC1", "ring", "C5H9"),
    Fragment("cyclopropyl", "[*:1]C1CC1", "ring", "C3H5"),
    Fragment("2-furanyl", "[*:1]c1ccco1", "ring", "2-Fur"),
    Fragment("3-furanyl", "[*:1]c1ccoc1", "ring", "3-Fur"),
    Fragment("2-thiophenyl", "[*:1]c1cccs1", "ring", "2-Thi"),
    Fragment("3-thiophenyl", "[*:1]c1ccsc1", "ring", "3-Thi"),
    # ---- common substituents
    Fragment("isopropyl", "[*:1]C(C)C", "substituent", "iPr"),
    Fragment("tert-butyl", "[*:1]C(C)(C)C", "substituent", "tBu"),
    Fragment("methoxy", "[*:1]OC", "substituent", "OMe"),
    Fragment("trifluoromethoxy", "[*:1]OC(F)(F)F", "substituent", "OCF3"),
    Fragment("acetyl", "[*:1]C(C)=O", "substituent", "Ac"),
    Fragment("vinyl", "[*:1]C=C", "substituent", "CH=CH2"),
    Fragment("ethynyl", "[*:1]C#C", "substituent", "C≡CH"),
    # ---- PFAS building blocks
    Fragment("trifluoromethyl", "[*:1]C(F)(F)F", "pfas", "CF3"),
    Fragment("difluoromethylene", "[*:1]C(F)(F)[*:2]", "pfas", "CF2", "one CF2 unit; cf2_chain(n) makes n"),
    Fragment("ether o", "[*:1]O[*:2]", "pfas", "-O-", "ether link of PFPE and GenX-type chains"),
    Fragment("perfluoroether unit", "[*:1]OC(F)(F)[*:2]", "pfas", "OCF2", "PFPE repeat unit"),
    Fragment("hfpo unit", "[*:1]C(F)(C(F)(F)F)C(F)(F)O[*:2]", "pfas", "HFPO",
             "-CF(CF3)CF2O-, the hexafluoropropylene oxide repeat unit"),
    Fragment("fluorotelomer linker", "[*:1]CC[*:2]", "pfas", "CH2CH2", "the -CH2CH2- of n:2 fluorotelomers"),
    Fragment("pfca head", "[*:1]C(F)(F)C(=O)O", "pfas", "CF2COOH", "perfluoroalkyl carboxylic acid head"),
    Fragment("pfca head anion", "[*:1]C(F)(F)C(=O)[O-]", "pfas", "CF2COO-"),
    Fragment("pfsa head", "[*:1]C(F)(F)S(=O)(=O)O", "pfas", "CF2SO3H", "perfluoroalkane sulfonic acid head"),
    Fragment("pfsa head anion", "[*:1]C(F)(F)S(=O)(=O)[O-]", "pfas", "CF2SO3-"),
    Fragment("perfluoroethyl", "[*:1]C(F)(F)C(F)(F)F", "pfas", "C2F5"),
    Fragment("perfluoropropyl", "[*:1]C(F)(F)C(F)(F)C(F)(F)F", "pfas", "C3F7"),
    Fragment("perfluorobutyl", "[*:1]" + "C(F)(F)" * 3 + "C(F)(F)F", "pfas", "C4F9"),
    Fragment("perfluorohexyl", "[*:1]" + "C(F)(F)" * 5 + "C(F)(F)F", "pfas", "C6F13"),
    Fragment("perfluorooctyl", "[*:1]" + "C(F)(F)" * 7 + "C(F)(F)F", "pfas", "C8F17"),
)

FRAGMENTS: dict[str, Fragment] = {f.name: f for f in _LIBRARY}

_ALIASES = {
    "me": "methyl", "et": "ethyl", "oh": "hydroxyl", "nh2": "amine", "amino": "amine", "cho": "carbonyl",
    "formyl": "carbonyl", "cooh": "carboxylic acid", "carboxyl": "carboxylic acid", "coo-": "carboxylate",
    "no2": "nitro", "cn": "nitrile", "cyano": "nitrile", "sh": "thiol", "so3h": "sulfonic acid",
    "so3-": "sulfonate", "ph": "phenyl", "benzene": "phenyl", "cyclohexane": "cyclohexyl",
    "pyridyl": "2-pyridyl", "pyridine": "2-pyridyl", "furanyl": "2-furanyl", "furan": "2-furanyl",
    "thiophenyl": "2-thiophenyl", "thienyl": "2-thiophenyl", "thiophene": "2-thiophenyl",
    "cyclopentane": "cyclopentyl", "cf3": "trifluoromethyl", "cf2": "difluoromethylene", "ether": "ether o",
    "ft linker": "fluorotelomer linker", "telomer linker": "fluorotelomer linker",
}


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("_", " "))


def cf2_chain(n: int) -> Fragment:
    """``-(CF2)n-`` with a hydrogen at the far end (``[*:2]``) where the chain continues."""
    if n < 1:
        raise ValueError("A (CF2)n chain needs n >= 1")
    return Fragment(f"(cf2){n} chain", "[*:1]" + "C(F)(F)" * n + "[*:2]", "pfas", f"(CF2){n}")


def perfluoroalkyl(n: int) -> Fragment:
    """The linear perfluoroalkyl group CnF(2n+1) (n = 1 is CF3)."""
    if n < 1:
        raise ValueError("A perfluoroalkyl group needs n >= 1")
    return Fragment(f"perfluoroalkyl c{n}f{2 * n + 1}", "[*:1]" + "C(F)(F)" * (n - 1) + "C(F)(F)F", "pfas",
                    f"C{n}F{2 * n + 1}")


def from_smiles(smiles: str, name: str = "", category: str = "group") -> Fragment:
    """A fragment typed by the user (the kit's SMILES button). Without a ``*`` the first atom of the SMILES is
    the attachment atom (it must have a hydrogen to give up), as in Spartan."""
    smiles = smiles.strip()
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Not a valid SMILES string: {smiles!r}")
    if not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        if mol.GetNumAtoms() == 0 or mol.GetAtomWithIdx(0).GetTotalNumHs() == 0:
            raise ValueError(f"{smiles!r}: the first atom has no hydrogen to replace; mark the attachment "
                             "point with [*:1]")
        rw = Chem.RWMol(mol)
        first = rw.GetAtomWithIdx(0)
        if first.GetNumExplicitHs():
            first.SetNumExplicitHs(first.GetNumExplicitHs() - 1)
        d = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(d).SetAtomMapNum(1)
        rw.AddBond(0, d, Chem.BondType.SINGLE)
        Chem.SanitizeMol(rw)
        smiles = Chem.MolToSmiles(rw)
    frag = Fragment(_key(name) if name else smiles, smiles, category, name or smiles)
    attachment_points(frag)  # validates
    return frag


def get(name: str | Fragment) -> Fragment:
    """A library fragment by name, label or alias (case ignored), or a builder written as text:
    ``(CF2)6`` -> ``cf2_chain(6)``, ``C6F13`` -> ``perfluoroalkyl(6)``."""
    if isinstance(name, Fragment):
        return name
    key = _key(name)
    if key in FRAGMENTS:
        return FRAGMENTS[key]
    if key in _ALIASES:
        return FRAGMENTS[_ALIASES[key]]
    for f in _LIBRARY:
        if f.label and f.label.lower() == key:
            return f
    m = re.fullmatch(r"\(cf2\)\s*(\d+)(?: chain)?", key)
    if m:
        return cf2_chain(int(m.group(1)))
    m = re.fullmatch(r"(?:perfluoroalkyl )?c(\d+)f(\d+)", key)
    if m and int(m.group(2)) == 2 * int(m.group(1)) + 1:
        return perfluoroalkyl(int(m.group(1)))
    raise KeyError(f"No fragment named {name!r}")


def by_category(category: str) -> list[Fragment]:
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    return [f for f in _LIBRARY if f.category == category]


def attachment_points(fragment: Fragment) -> tuple[Chem.Mol, int, int | None]:
    """Parse ``fragment``: (molecule with its dummy atoms, index of the ``[*:1]`` dummy, index of the
    ``[*:2]`` dummy or None). A single unnumbered ``*`` counts as ``[*:1]``. Raises ValueError when the
    attachment is missing or ambiguous."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(fragment.smiles)
    if mol is None:
        raise ValueError(f"Fragment {fragment.name!r}: not a valid SMILES string: {fragment.smiles!r}")
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    by_map: dict[int, int] = {}
    for a in dummies:
        if a.GetAtomMapNum() in by_map:
            raise ValueError(f"Fragment {fragment.name!r}: two dummy atoms numbered {a.GetAtomMapNum()}")
        by_map[a.GetAtomMapNum()] = a.GetIdx()
    if 1 not in by_map and len(dummies) == 1 and 0 in by_map:
        by_map = {1: by_map[0]}
    if 1 not in by_map:
        raise ValueError(f"Fragment {fragment.name!r} has no attachment point [*:1]")
    if set(by_map) - {1, 2}:
        raise ValueError(f"Fragment {fragment.name!r}: only [*:1] (attachment) and [*:2] (chain end) are allowed")
    for idx in by_map.values():
        if mol.GetAtomWithIdx(idx).GetDegree() != 1:
            raise ValueError(f"Fragment {fragment.name!r}: each dummy atom must have exactly one neighbour")
    exit_ = by_map.get(2)
    if exit_ is not None and mol.GetBondBetweenAtoms(exit_, mol.GetAtomWithIdx(exit_).GetNeighbors()[0].GetIdx()) \
            .GetBondType() != Chem.BondType.SINGLE:
        raise ValueError(f"Fragment {fragment.name!r}: the chain end [*:2] must be a single bond")
    return mol, by_map[1], exit_
