"""Standardization and protonation.

* ``standardize``: RDKit cleanup, keep the largest organic fragment (strips counter-ions such as K+ in
  potassium PFOS), and neutralize, so every input starts from the same parent structure.
* ``protonate``: assign the dominant charge state at a given pH. The built-in rule table is small,
  transparent and PFAS-aware (e.g. perfluoroalkane sulfonamides are acidic, pKa about 6.5). If
  ``dimorphite-dl`` is installed it can be used instead (``method="dimorphite"``).
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class SiteRule:
    name: str
    smarts: str          # first mapped atom (:1) is the site that gains/loses H
    pka: float
    kind: str            # "acid" (loses H+ when pH > pKa) or "base" (gains H+ when pH < pKa)


# Ordered roughly from strongest acid to weakest base. Values are typical literature pKa's;
# they only need to be on the right side of the working pH.
RULES: tuple[SiteRule, ...] = (
    # The Uncharger turns halide anions into HX; these put them back (HCl -7, HBr -9, HI -10, HF 3.2)
    SiteRule("hydrohalic acid", "[Cl,Br,I;H1;D0;+0:1]", -7.0, "acid"),
    SiteRule("hydrofluoric acid", "[F;H1;D0;+0:1]", 3.2, "acid"),
    SiteRule("sulfate monoester", "[OX2H1:1][SX4](=O)(=O)[OX2]", -2.0, "acid"),
    SiteRule("sulfonic acid", "[OX2H1:1][SX4](=O)(=O)[#6]", -1.5, "acid"),
    SiteRule("phosphate/phosphonate (1st OH)", "[OX2H1:1][PX4](=O)", 1.8, "acid"),
    SiteRule("carboxylic acid", "[OX2H1:1][CX3](=O)", 4.0, "acid"),
    SiteRule("perfluoroalkyl sulfonamide N-H", "[NX3;H2,H1;!$(N[#6]=O):1][SX4](=O)(=O)C(F)(F)", 6.5, "acid"),
    SiteRule("aliphatic amine", "[NX3;H3,H2,H1,H0;+0;!$(N[a]);!$(N[#6,#16,#15]=[O,S,N]);!$(N[SX4]);!$(N-[#6](F));!$(N#*);!$(N=*):1]",
             10.0, "base"),
    SiteRule("amidine/guanidine", "[NX3;H2,H1;+0:1][CX3]=[NX2]", 12.0, "base"),
)


def standardize(mol: Chem.Mol, keep_largest_fragment: bool = True, neutralize: bool = True) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol = Chem.RemoveHs(mol, sanitize=True)
    mol = rdMolStandardize.Cleanup(mol)
    if keep_largest_fragment and len(Chem.GetMolFrags(mol)) > 1:
        mol = rdMolStandardize.FragmentParent(mol)
    if neutralize:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return mol


def _apply_rules(mol: Chem.Mol, ph: float) -> tuple[Chem.Mol, list[str]]:
    rw = Chem.RWMol(mol)
    changes = []
    done: set[int] = set()
    for rule in RULES:
        patt = Chem.MolFromSmarts(rule.smarts)
        site_idx = next((a.GetIdx() for a in patt.GetAtoms() if a.GetAtomMapNum() == 1), 0)
        for match in rw.GetSubstructMatches(patt):
            idx = match[site_idx]
            if idx in done:
                continue
            atom = rw.GetAtomWithIdx(idx)
            if rule.kind == "acid" and ph > rule.pka and atom.GetTotalNumHs() > 0:
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
                atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
                atom.SetNoImplicit(True)
                changes.append(f"deprotonated {rule.name} (pKa~{rule.pka:g})")
                done.add(idx)
            elif rule.kind == "base" and ph < rule.pka:
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                atom.SetNoImplicit(True)
                changes.append(f"protonated {rule.name} (pKa~{rule.pka:g})")
                done.add(idx)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out, changes


def protonate(mol: Chem.Mol, ph: float = 7.0, method: str = "rules") -> tuple[Chem.Mol, list[str]]:
    """Return (molecule in its dominant state at ``ph``, list of human-readable notes)."""
    if method == "dimorphite":
        try:
            from dimorphite_dl import protonate_smiles
        except ImportError:
            method = "rules"
        else:
            variants = protonate_smiles(Chem.MolToSmiles(mol), ph_min=ph, ph_max=ph, precision=0.5)
            if variants:
                out = Chem.MolFromSmiles(variants[0])
                notes = [f"dimorphite-dl: {len(variants)} state(s) at pH {ph:g}, using {variants[0]}"]
                if out is not None:
                    return out, notes
    return _apply_rules(mol, ph)


def formal_charge(mol: Chem.Mol) -> int:
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def prepare(mol: Chem.Mol, ph: float | None = 7.0, keep_largest_fragment: bool = True,
            method: str = "rules") -> tuple[Chem.Mol, list[str]]:
    """standardize -> protonate (if ``ph`` is not None) -> add explicit hydrogens."""
    notes = []
    std = standardize(mol, keep_largest_fragment=keep_largest_fragment, neutralize=ph is not None)
    if Chem.MolToSmiles(std) != Chem.MolToSmiles(Chem.RemoveHs(mol)):
        notes.append(f"standardized to {Chem.MolToSmiles(std)}")
    if ph is not None:
        std, changes = protonate(std, ph, method)
        notes += changes
    return Chem.AddHs(std), notes
