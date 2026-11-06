"""Standardization and protonation.

* ``standardize``: RDKit cleanup, keep the largest fragment (strips counter-ions such as K+ in
  potassium PFOS), and neutralize, so every input starts from the same parent structure.
* ``protonate``: assign the dominant charge state at a given pH. The built-in rule table is small,
  transparent and PFAS-aware (e.g. perfluoroalkane sulfonamides are acidic, pKa about 6.5). If
  ``dimorphite-dl`` is installed it can be used instead (``method="dimorphite"``).
* ``spin_multiplicity``: the multiplicity to compute a structure with, from its unpaired electrons.

Salts, mixtures and metals (the conservative choices; ``prepare`` notes each one it makes):

* Bonds to alkali and alkaline-earth metals (Na-O in a sodium carboxylate drawn covalently) are ionic:
  they are disconnected, as RDKit's ``MetalDisconnector`` does. Bonds to every other metal (Sn-C, Hg-O,
  Fe-C) are kept as drawn, so an organotin or organomercury compound is computed whole (xTB covers
  these elements), not as a cation with its ligand silently removed.
* Of several fragments the largest is kept. Removed fragments are named in the notes ("removed Na+
  (counter-ion)"). Bondless Al, Fe, Cu and Zn cations (``SALT_METALS``) are counter-ions too when the
  record is a salt of an organic ion (fosetyl-Al: one ethyl phosphonate anion is computed); see
  ``salt_counterion_problem`` for the test. A removed fragment holding any other metal, or a salt metal
  next to fragments that are not counter-ions (a carbonyl complex, a cobalamin, a phosphine complex or a
  ferrocene drawn as ions), raises ``UnsupportedStructure``: computing the ligand alone would be a
  different molecule.
* ``inchikey``: the standard InChIKey of a species, for joining results to other tables by structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

#: Alkali and alkaline-earth metals: their bonds are treated as ionic, and their ions as counter-ions.
COUNTERION_METALS = frozenset({3, 11, 19, 37, 55, 87, 4, 12, 20, 38, 56, 88})
#: Al, Fe, Cu, Zn: a bondless cation of these is a salt counter-ion when the rest of the record is an organic
#: ion and counter-ions (``salt_counterion_problem``). Their bonds are never disconnected: an Al-O, Zn-C or
#: Fe-C bond drawn in the record keeps the molecule whole, as for every metal outside COUNTERION_METALS.
SALT_METALS = frozenset({13, 26, 29, 30})
#: Every metal (metalloids B, Si, Ge, As, Sb, Te are not): Al, Ga, In, Sn, Tl, Pb, Bi, Po, the
#: transition metals, lanthanides and actinides, and the alkali/alkaline-earth metals.
METALS = frozenset({3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 84, 87, 88}
                   | set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 104)))


class UnsupportedStructure(ValueError):
    """The structure cannot be computed faithfully (e.g. a metal complex drawn as separate ions)."""


def _ionic_metal_disconnector() -> rdMolStandardize.MetalDisconnector:
    """RDKit's MetalDisconnector restricted to alkali and alkaline-earth metals."""
    md = rdMolStandardize.MetalDisconnector()
    metals = ",".join(f"#{z}" for z in sorted(COUNTERION_METALS))
    md.SetMetalNof(Chem.MolFromSmarts(f"[{metals}]~[#7,#8,F]"))
    md.SetMetalNon(Chem.MolFromSmarts(f"[{metals}]~[#5,#6,#14,#15,#16,#17,#33,#34,#35,#51,#52,#53,#85]"))
    return md


def _fragment_label(frag: Chem.Mol, max_len: int = 40) -> str:
    """SMILES of a fragment, or its formula when the SMILES is long (an error message is one table cell)."""
    from rdkit.Chem import rdMolDescriptors

    with rdBase.BlockLogs():
        smi = Chem.MolToSmiles(Chem.RemoveHs(frag, sanitize=False))
        if len(smi) <= max_len:
            return smi
        frag = Chem.Mol(frag)
        frag.UpdatePropertyCache(strict=False)
        return rdMolDescriptors.CalcMolFormula(frag)


def _describe_removed(labels: list[str]) -> str:
    """'[Na+]', '2 x [Cl-]', ... (fragment SMILES) for the notes."""
    counts = Counter(labels)
    return ", ".join(f"{n} x {s}" if n > 1 else s for s, n in counts.items())


def _metal_ion(frag: Chem.Mol, metals: frozenset[int]) -> bool:
    """A single bondless atom of ``metals`` with a positive charge (Al3+, Zn2+, ...)."""
    if frag.GetNumAtoms() != 1:
        return False
    atom = frag.GetAtomWithIdx(0)
    return atom.GetAtomicNum() in metals and atom.GetFormalCharge() > 0


def _carbanion(frag: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == 6 and a.GetFormalCharge() < 0 for a in frag.GetAtoms())


def salt_counterion_problem(parent: Chem.Mol, removed: list[Chem.Mol]) -> str:
    """Why the ``SALT_METALS`` cations among ``removed`` are not plain counter-ions of ``parent`` ("" if they are).

    They are when the kept fragment is an organic ion whose charge does not sit on carbon (a carboxylate,
    phosphonate, sulfonate, phenolate or ammonium, not a cyclopentadienide or cyanide ligand) and every other
    removed fragment is a counter-ion too: a metal cation, a copy of the kept ion (salt stoichiometry), an
    inorganic ion or water of hydration, or another organic ion of that kind. A neutral organic fragment
    (CO, a phosphine, a corrin) next to the metal makes the record a metal complex drawn without its bonds."""
    if not any(a.GetAtomicNum() == 6 for a in parent.GetAtoms()):
        return "the kept fragment is inorganic"
    if sum(a.GetFormalCharge() for a in parent.GetAtoms()) == 0:
        return "the kept fragment is neutral, so the metal is not its counter-ion"
    if _carbanion(parent):
        return "the kept fragment is a carbanion (a ligand, e.g. cyclopentadienide or cyanide)"
    key = _fragment_label(parent, max_len=10**6)
    for frag in removed:
        if _metal_ion(frag, SALT_METALS | COUNTERION_METALS) or _fragment_label(frag, max_len=10**6) == key:
            continue
        charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        organic = any(a.GetAtomicNum() == 6 for a in frag.GetAtoms())
        water = Chem.MolToSmiles(Chem.RemoveHs(frag, sanitize=False)) == "O"
        if any(a.GetAtomicNum() in METALS for a in frag.GetAtoms()):
            return f"{_fragment_label(frag)} holds another metal"
        if not organic and (charge != 0 or water):
            continue                       # halide, sulfate, nitrate, hydroxide, water of hydration
        if organic and charge != 0 and not _carbanion(frag):
            continue                       # a second organic counter-ion
        return f"{_fragment_label(frag)} is not a counter-ion (a ligand)"
    return ""


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


def standardize(mol: Chem.Mol, keep_largest_fragment: bool = True, neutralize: bool = True,
                info: dict | None = None) -> Chem.Mol:
    """Cleanup (ionic metal bonds disconnected, functional groups normalized, reionized), the largest
    fragment, then neutralization. ``info`` (optional dict) receives ``removed_fragments`` (SMILES of the
    fragments removed) and ``n_fragments``. Raises ``UnsupportedStructure`` if a removed fragment holds
    a metal other than an alkali or alkaline-earth counter-ion."""
    mol = Chem.Mol(mol)
    placeholders = sorted({f"{a.GetSymbol()}{a.GetIdx() + 1}" if a.GetAtomicNum() else "*"
                           for a in mol.GetAtoms() if a.GetAtomicNum() == 0 or a.HasQuery()})
    if placeholders:   # R groups, "*", atom lists (M  ALS): ETKDG aborts on them with an RDKit invariant violation
        raise UnsupportedStructure(f"not a definite structure: query or placeholder atom(s) {', '.join(placeholders)} "
                                   "(R group, '*' or an atom list in the record)")
    with rdBase.BlockLogs():         # "not removing hydrogen atom without neighbors" for H- / H+ fragments
        mol = Chem.RemoveHs(mol, sanitize=True)
    # rdMolStandardize.Cleanup, but with the metal disconnector limited to ionic (group 1/2) metal bonds
    mol = _ionic_metal_disconnector().Disconnect(mol)
    mol = rdMolStandardize.Normalize(mol)
    mol = rdMolStandardize.Reionize(mol)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    removed: list[Chem.Mol] = []
    if keep_largest_fragment and len(frags) > 1:
        parent = rdMolStandardize.FragmentParent(mol, skipStandardize=True)   # cleaned above, our way
        labels = [_fragment_label(f, max_len=10**6) for f in frags]
        key = _fragment_label(parent, max_len=10**6)
        keep = labels.index(key) if key in labels else max(range(len(frags)), key=lambda i: frags[i].GetNumAtoms())
        removed = [f for i, f in enumerate(frags) if i != keep]
        salt_problem = ""
        if any(_metal_ion(f, SALT_METALS) for f in removed):
            salt_problem = salt_counterion_problem(parent, removed)
        metals = sorted({a.GetSymbol() for f in removed for a in f.GetAtoms()
                         if a.GetAtomicNum() in METALS
                         and not (a.GetAtomicNum() in COUNTERION_METALS and f.GetNumAtoms() == 1)
                         and not (_metal_ion(f, SALT_METALS) and not salt_problem)})
        if metals:
            raise UnsupportedStructure(
                f"metal complex: keeping the largest fragment ({_fragment_label(parent)}) would remove "
                f"{', '.join(metals)} ({_describe_removed([_fragment_label(f) for f in removed])})"
                + (f": {salt_problem}" if salt_problem else "") + "; not computed. "
                "A complex drawn as separate ions (or without its metal-ligand bonds) is a different molecule "
                "without its metal; draw the metal-ligand bonds to compute it whole")
        mol = parent
    if info is not None:
        info["n_fragments"] = len(frags)
        info["removed_fragments"] = [_fragment_label(f, max_len=10**6) for f in removed]
    if neutralize:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return mol


#: A basic site within this many bonds of an N+ (one protonated by these rules, or a quaternary ammonium) is
#: left neutral: the second pKa of piperazine (5.6), ethylenediamine (6.9) or a guanidine's other NH2 is at or
#: below pH 7, so these get one proton. 1,3-diamines (4 bonds apart, second pKa ~ 9) still get two.
BASE_EXCLUSION_BONDS = 3


def _ammonium(atom: Chem.Atom) -> bool:
    """An sp3 N+ (ammonium), not the N+ of a nitro group, an N-oxide or an aromatic cation."""
    return (atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 and not atom.GetIsAromatic()
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
            and all(nb.GetFormalCharge() >= 0 for nb in atom.GetNeighbors()))


def _apply_rules(mol: Chem.Mol, ph: float) -> tuple[Chem.Mol, list[str]]:
    rw = Chem.RWMol(mol)
    changes = []
    done: set[int] = set()
    dist = Chem.GetDistanceMatrix(mol)
    cations = {a.GetIdx() for a in mol.GetAtoms() if _ammonium(a)}
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
                done.add(idx)
                if any(dist[idx][j] <= BASE_EXCLUSION_BONDS for j in cations):
                    changes.append(f"left {rule.name} neutral (within {BASE_EXCLUSION_BONDS} bonds of an N+: one "
                                   "proton for basic sites that close, as in piperazine or a 1,2-diamine)")
                    continue
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                atom.SetNoImplicit(True)
                changes.append(f"protonated {rule.name} (pKa~{rule.pka:g})")
                cations.add(idx)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    counts = Counter(changes)     # one note per rule: "deprotonated 2 x phosphate/phosphonate ..."
    return out, [c if n == 1 else c.replace("left ", f"left {n} x ", 1) if c.startswith("left ")
                 else c.replace("protonated ", f"protonated {n} x ", 1) for c, n in counts.items()]


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


def inchikey(mol: Chem.Mol | None) -> tuple[str, str]:
    """(standard InChIKey, note) of ``mol`` as it is: charges, fragments and stereo as drawn, so an anion
    and its acid differ in the protonation flag (the last letter: -M for one proton removed, -N neutral). ``("",
    why)`` when RDKit has no InChI support or InChI cannot represent the species (the note says so)."""
    if mol is None:
        return "", "no structure"
    from rdkit.Chem import inchi as rd_inchi

    if not rd_inchi.INCHI_AVAILABLE:
        return "", "RDKit was built without InChI support"
    try:
        with rdBase.BlockLogs():
            text = rd_inchi.MolToInchi(mol)
            key = rd_inchi.InchiToInchiKey(text) if text else ""
    except Exception as exc:  # noqa: BLE001 - InChI's reason, as a note
        return "", f"InChI failed ({type(exc).__name__}: {exc})"
    return (key, "") if key else ("", "InChI could not represent this structure")


def unpaired_electrons(mol: Chem.Mol) -> int:
    return sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


def electron_count(mol: Chem.Mol, charge: int | None = None) -> int:
    """Electrons of the molecule with its explicit and implicit hydrogens at total ``charge``."""
    z = sum(a.GetAtomicNum() + a.GetTotalNumHs() for a in mol.GetAtoms())
    return z - (formal_charge(mol) if charge is None else int(charge))


def spin_multiplicity(mol: Chem.Mol, charge: int, requested: int | None = None) -> tuple[int, str]:
    """(multiplicity, note) for ``mol`` at total ``charge``.

    ``requested`` (an explicit multiplicity from the input) wins if its parity fits the electron count.
    Otherwise radicals drawn in the structure (RDKit's unpaired electrons) give the high-spin
    multiplicity 2S+1 = n + 1, with a note, because database records with radicals are often drawing
    errors (a missing H or charge). An odd electron count without drawn radicals is a doublet. Raises
    ``UnsupportedStructure`` when the requested multiplicity cannot fit the electron count."""
    n_el = electron_count(mol, charge)
    n_rad = unpaired_electrons(mol)
    if requested is not None and requested >= 1:
        if (requested - 1) % 2 != n_el % 2:
            raise UnsupportedStructure(f"multiplicity {requested} is impossible with {n_el} electrons "
                                       f"(charge {charge:+d})")
        return requested, ""
    if n_rad:
        mult = n_rad + 1
        if (mult - 1) % 2 != n_el % 2:     # drawn radicals and the charge disagree: say so, use the minimum
            mult = 1 + n_el % 2
        return mult, (f"open shell: {n_rad} unpaired electron(s) drawn in the structure; computed with "
                      f"multiplicity {mult} (check the record: radicals in databases are often a missing H or "
                      "charge)")
    if n_el % 2:
        return 2, f"odd electron count ({n_el}): computed as a doublet (multiplicity 2)"
    return 1, ""


def prepare(mol: Chem.Mol, ph: float | None = 7.0, keep_largest_fragment: bool = True,
            method: str = "rules", info: dict | None = None) -> tuple[Chem.Mol, list[str]]:
    """standardize -> protonate (if ``ph`` is not None) -> add explicit hydrogens.

    The notes say what was done: fragments removed (by name), neutralization, protonation, and the
    charge as drawn versus as computed. ``info`` (optional dict) also receives ``input_charge``,
    ``removed_fragments`` and ``n_fragments``. Raises ``UnsupportedStructure`` (see ``standardize``)."""
    notes = []
    std_info: dict = {}
    with rdBase.BlockLogs():
        drawn = Chem.MolToSmiles(Chem.RemoveHs(mol))
    q_in = formal_charge(mol)
    std = standardize(mol, keep_largest_fragment=keep_largest_fragment, neutralize=ph is not None, info=std_info)
    removed = std_info.get("removed_fragments") or []
    if removed:
        notes.append(f"kept the largest of {std_info['n_fragments']} fragments; removed "
                     f"{_describe_removed(removed)}")
    elif std_info.get("n_fragments", 1) > 1:
        notes.append(f"{std_info['n_fragments']} fragments computed together (keep largest fragment is off)")
    if Chem.MolToSmiles(std) != drawn:
        notes.append(f"standardized to {Chem.MolToSmiles(std)}")
    if ph is not None:
        std, changes = protonate(std, ph, method)
        notes += changes
    q = formal_charge(std)
    if q != q_in:
        why = "counter-ions removed" if removed else ""
        if ph is not None:
            why += (", then " if why else "") + f"neutralized and protonated for pH {ph:g}"
        notes.append(f"charge {q_in:+d} as drawn, {q:+d} computed ({why})")
    if info is not None:
        info.update(std_info, input_charge=q_in)
    return Chem.AddHs(std), notes
