"""Standardization and protonation.

* ``standardize``: RDKit cleanup, the build filters (``FragmentStrip``: water, acids and named counter-ions,
  Settings > Build defaults), keep the largest fragment (strips counter-ions such as K+ in potassium PFOS), and
  neutralize, so every input starts from the same parent structure.
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

import functools
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


#: Counter-ions a build can strip by name before the largest-fragment rule (``FragmentStrip``, ``BuildConfig.
#: strip_counterions``; Settings > Build defaults). A fragment is the named ion when its neutral parent (RDKit's
#: Uncharger: HCl for chloride, triethylamine for triethylammonium, Na+ stays Na+) is the ion's. Only alkali and
#: alkaline-earth metals are here: a fragment with another metal is for the build's metal-complex rule to decide.
NAMED_COUNTERIONS: dict[str, str] = {
    "lithium": "[Li+]", "sodium": "[Na+]", "potassium": "[K+]", "rubidium": "[Rb+]", "cesium": "[Cs+]",
    "magnesium": "[Mg+2]", "calcium": "[Ca+2]", "strontium": "[Sr+2]", "barium": "[Ba+2]",
    "ammonium": "[NH4+]", "tetramethylammonium": "C[N+](C)(C)C", "tetraethylammonium": "CC[N+](CC)(CC)CC",
    "tetrabutylammonium": "CCCC[N+](CCCC)(CCCC)CCCC", "triethylammonium": "CC[NH+](CC)CC",
    "ethanolammonium": "OCC[NH3+]", "diethanolammonium": "OCC[NH2+]CCO", "triethanolammonium": "OCC[NH+](CCO)CCO",
    "fluoride": "[F-]", "chloride": "[Cl-]", "bromide": "[Br-]", "iodide": "[I-]", "hydroxide": "[OH-]",
    "nitrate": "[O-][N+](=O)[O-]", "sulfate": "[O-]S(=O)(=O)[O-]", "hydrogensulfate": "OS(=O)(=O)[O-]",
    "phosphate": "[O-]P(=O)([O-])[O-]", "perchlorate": "[O-]Cl(=O)(=O)=O", "tetrafluoroborate": "F[B-](F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F", "acetate": "CC(=O)[O-]", "formate": "O=C[O-]",
    "trifluoroacetate": "O=C([O-])C(F)(F)F", "mesylate": "CS(=O)(=O)[O-]", "tosylate": "Cc1ccc(S(=O)(=O)[O-])cc1",
}


def counterion_names(value, strict: bool = True) -> tuple[str, ...]:
    """The counter-ion names in ``value`` (a comma- or space-separated string, or a list), lowercase, sorted, each once.
    An unknown name raises ``ValueError`` (``strict``) or is left out."""
    import re

    parts = re.split(r"[,;+\s]+", value) if isinstance(value, str) else [str(v) for v in (value or ())]
    names = {p.strip().lower() for p in parts if p and p.strip()}
    unknown = sorted(n for n in names if n not in NAMED_COUNTERIONS)
    if unknown and strict:
        raise ValueError(f"unknown counter-ion(s) {', '.join(unknown)} (known: {', '.join(NAMED_COUNTERIONS)})")
    return tuple(sorted(names - set(unknown)))


def canonical_counterions(value, strict: bool = True) -> str:
    """``counterion_names`` as the one string a run stores ("potassium,sodium"), so equal choices compare equal."""
    return ",".join(counterion_names(value, strict))


@functools.cache
def _neutral_key(smiles: str) -> str:
    """Canonical SMILES of the neutral parent of ``smiles`` (one fragment): the key named counter-ions match by."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(mol))


@dataclass(frozen=True)
class FragmentStrip:
    """Fragments a build removes before the largest-fragment rule (SETTINGS-1; Spartan's Import Filters, which strip
    water and HCl from SD files): water, small neutral inorganic acids (HCl, HBr, HI, HF and others: ``import_options.
    fragment_kind``), and the counter-ions named in ``counterions`` (``NAMED_COUNTERIONS``). Science settings
    (``BuildConfig.strip_water``, ``strip_acids``, ``strip_counterions``). They matter where the largest-fragment rule
    would not remove the fragment: with *keep largest fragment* off (a hydrate's ion pair is computed without its
    water), or where the counter-ion is the larger ion (tetrabutylammonium trifluoroacetate)."""

    water: bool = False
    acids: bool = False
    counterions: tuple[str, ...] = ()

    @classmethod
    def of(cls, cfg, strict: bool = True) -> FragmentStrip:
        """From a ``BuildConfig`` (or anything with its three fields); unknown counter-ion names raise (``strict``)."""
        return cls(water=bool(getattr(cfg, "strip_water", False)), acids=bool(getattr(cfg, "strip_acids", False)),
                   counterions=counterion_names(getattr(cfg, "strip_counterions", "") or "", strict))

    def __bool__(self) -> bool:
        return self.water or self.acids or bool(self.counterions)

    def label(self) -> str:
        """'water, acids, sodium' (what is stripped), for notes."""
        return ", ".join([*(["water"] if self.water else []), *(["acids"] if self.acids else []), *self.counterions])

    def why(self, mol: Chem.Mol, atoms: tuple[int, ...]) -> str:
        """Why the fragment of ``mol`` made of ``atoms`` is stripped ("water", "acid", a counter-ion name), else ""."""
        from .import_options import fragment_kind

        kind = fragment_kind(mol, atoms) if (self.water or self.acids) else ""
        if (kind == "water" and self.water) or (kind == "acid" and self.acids):
            return kind
        if self.counterions:
            with rdBase.BlockLogs():
                key = _neutral_key(Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms)))
            for name in self.counterions:
                if key == _neutral_key(NAMED_COUNTERIONS[name]):
                    return name
        return ""


def _strip_fragments(mol: Chem.Mol, strip: FragmentStrip) -> tuple[Chem.Mol, list[tuple[str, str]], str]:
    """(``mol`` without the fragments ``strip`` removes, (SMILES, reason) of each, a note when every fragment matched
    and the largest of them was kept)."""
    frags = list(Chem.GetMolFrags(mol))
    if len(frags) < 2 or not strip:
        return mol, [], ""
    why = {i: strip.why(mol, f) for i, f in enumerate(frags)}
    drop = [i for i, w in why.items() if w]
    note = ""
    if len(drop) == len(frags):          # never remove everything: keep the largest of them
        keep = max(drop, key=lambda i: (len(frags[i]), -i))
        drop.remove(keep)
        note = (f"every fragment matched the strip filters ({strip.label()}); kept the largest, "
                f"{_fragment_label(_submol(mol, frags[keep]))}")
    if not drop:
        return mol, [], note
    labels = [(_fragment_label(_submol(mol, frags[i]), max_len=10**6), why[i]) for i in drop]
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for i in drop:
        for a in frags[i]:
            rw.RemoveAtom(a)
    rw.CommitBatchEdit()
    out = rw.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(out)
    return out, labels, note


def _submol(mol: Chem.Mol, atoms: tuple[int, ...]) -> Chem.Mol:
    """The fragment of ``mol`` made of ``atoms``, as a molecule (for its label)."""
    with rdBase.BlockLogs():
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms))
        return Chem.MolFromSmiles(smi, sanitize=False) or Chem.Mol()


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
                info: dict | None = None, strip: FragmentStrip | None = None) -> Chem.Mol:
    """Cleanup (ionic metal bonds disconnected, functional groups normalized, reionized), the fragments ``strip``
    removes (water, acids, named counter-ions; ``FragmentStrip``), the largest fragment, then neutralization. ``info``
    (optional dict) receives ``removed_fragments`` (SMILES of the fragments removed, the stripped ones first),
    ``stripped_fragments`` ((SMILES, reason) of those ``strip`` removed), ``strip_note`` and ``n_fragments`` (as
    drawn). Raises ``UnsupportedStructure`` if a fragment the largest-fragment rule removes holds a metal other than
    an alkali or alkaline-earth counter-ion."""
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
    n_drawn = len(Chem.GetMolFrags(mol))
    stripped: list[tuple[str, str]] = []
    strip_note = ""
    if strip:                       # before the largest-fragment rule (Spartan's import filters)
        mol, stripped, strip_note = _strip_fragments(mol, strip)
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
        info["n_fragments"] = n_drawn
        info["stripped_fragments"] = stripped
        info["strip_note"] = strip_note
        info["removed_fragments"] = [s for s, _why in stripped] + [_fragment_label(f, max_len=10**6) for f in removed]
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


@functools.cache
def _rule_pattern(smarts: str) -> tuple[Chem.Mol, int]:
    """The compiled SMARTS of a rule and the index of its site atom (``:1``), compiled once per process: compiling
    the table took most of the protonation time of a 14,000-molecule review (``chem.advisor``)."""
    patt = Chem.MolFromSmarts(smarts)
    return patt, next((a.GetIdx() for a in patt.GetAtoms() if a.GetAtomMapNum() == 1), 0)


_N_PLUS = Chem.MolFromSmarts("[N+]")


def _apply_rules(mol: Chem.Mol, ph: float) -> tuple[Chem.Mol, list[str]]:
    rw = Chem.RWMol(mol)
    changes = []
    done: set[int] = set()
    # the distance matrix and the ammonium scan only matter for base sites near an N+: built when first needed
    # (most molecules have no N+, and a 14,000-molecule review spent most of its protonation time here)
    dist_cache: list = []

    def dist(i: int, j: int) -> float:
        if not dist_cache:
            dist_cache.append(Chem.GetDistanceMatrix(mol))
        return dist_cache[0][i][j]

    cations = ({a.GetIdx() for a in mol.GetAtoms() if _ammonium(a)} if mol.HasSubstructMatch(_N_PLUS) else set())
    for rule in RULES:
        patt, site_idx = _rule_pattern(rule.smarts)
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
                if any(dist(idx, j) <= BASE_EXCLUSION_BONDS for j in cations):
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
            method: str = "rules", info: dict | None = None,
            strip: FragmentStrip | None = None) -> tuple[Chem.Mol, list[str]]:
    """standardize (with the build's ``strip`` filters) -> protonate (if ``ph`` is not None) -> add explicit hydrogens.

    The notes say what was done: fragments stripped and removed (by name), neutralization, protonation, and the
    charge as drawn versus as computed. ``info`` (optional dict) also receives ``input_charge``,
    ``removed_fragments``, ``stripped_fragments`` and ``n_fragments``. Raises ``UnsupportedStructure`` (see
    ``standardize``)."""
    notes = []
    std_info: dict = {}
    with rdBase.BlockLogs():
        drawn = Chem.MolToSmiles(Chem.RemoveHs(mol))
    q_in = formal_charge(mol)
    std = standardize(mol, keep_largest_fragment=keep_largest_fragment, neutralize=ph is not None, info=std_info,
                      strip=strip)
    removed = std_info.get("removed_fragments") or []
    stripped = std_info.get("stripped_fragments") or []
    if stripped:
        counts = Counter(f"{s} ({why})" for s, why in stripped)
        notes.append("stripped before the largest-fragment rule (build filters): "
                     + ", ".join(f"{n} x {s}" if n > 1 else s for s, n in counts.items()))
    if std_info.get("strip_note"):
        notes.append(std_info["strip_note"])
    left = std_info.get("n_fragments", 1) - len(stripped)
    largest = removed[len(stripped):]
    if largest:
        notes.append(f"kept the largest of {left} fragments; removed {_describe_removed(largest)}")
    elif left > 1:
        notes.append(f"{left} fragments computed together (keep largest fragment is off)")
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
