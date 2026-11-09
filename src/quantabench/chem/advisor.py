"""The model advisor (ADV-1): a review of the molecules about to be built, before the run starts.

    review(inputs, cfg, *, available=None, max_ids=MAX_IDS) -> Review
    profile_batch(inputs, cfg) -> (Profile, molecules)        # the batch alone
    Review.text(), Review.header(), Review.to_dict()           # the CLI report, the build header, --json

What a review holds:

- **The profile of the batch**, each count with the molecule ids behind it (the first ``max_ids``): elements; the
  charge class at the run's pH (neutral, anion, cation, zwitterion) and the net charges; open shells; PFAS by the
  OECD 2021 structural definition and the perfluoroalkyl chain length; functional groups by SMARTS (``GROUPS``);
  pure hydrocarbons; heavy-atom and rotatable-bond distributions; macrocycles; undefined stereocentres;
  multi-fragment records (salts, counter-ions); isotopes; metals and other unusual elements.
- **Method coverage**: for each method, basis set and CREST level in ``method_guide.COVERAGE``, how many molecules
  it can take (elements, charge, spin) and whether it is installed here.
- **Findings**: ``Finding(level, topic, ids, message, recommendation, changes, why, guide)``. ``changes`` are
  ``BuildConfig`` fields to set; ``guide`` are methods-guide entries to read. Every rule rests on a measured
  result (docs/VALIDATION.md: the DFT conformer test, RANK-1, RANK-1b) or on a method's own documentation, named in
  ``evidence``; ``rank_verdict`` holds the measured conformer-ranking table.
- **Presets**: "Fast screen", "Recommended" and "Careful", each a ``BuildConfig`` with a plain-language rationale
  and an estimated cost on this machine (``estimate_cost``). A preset never uses a method outside a molecule's
  coverage.

Speed. The pipeline's ``standardize.prepare`` takes about 1 ms per molecule, mostly RDKit's normalization: 16 s for
14,000 molecules. The profile takes it only for records where it can change the species: several fragments, formal
charges, radicals, metals, Dimorphite-DL. Every other record (a neutral, single-fragment, metal-free structure,
most of a CompTox list) gets the same protonation rules (``standardize.protonate``) on the drawn structure, which
is what ``prepare`` computes for it. Molecules are profiled once per process for the same pH and protonation
settings (``_CACHE``), so a second review after "Apply" is instant, and only the first ``max_ids`` ids are kept per
finding. The GUI runs the review on a worker thread.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from . import method_guide as guide
from . import standardize

MAX_IDS = 25
LEVELS = ("problem", "warning", "info")
LEVEL_ORDER = {lv: i for i, lv in enumerate(LEVELS)}

#: Elements that need no comment in an organic batch; others are "unusual" (metals are listed apart)
COMMON_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})

EVIDENCE_DFT_TEST = "docs/VALIDATION.md, DFT conformer test"
EVIDENCE_RANK1 = "docs/VALIDATION.md, RANK-1"
EVIDENCE_RANK1B = "docs/VALIDATION.md, RANK-1b"
EVIDENCE_VAL3 = "docs/VALIDATION.md, VAL-3"

#: (key, label, SMARTS) of the functional groups in the profile; matched on the species the run computes
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("carboxylic", "carboxylic acid / carboxylate", "[CX3](=O)[OX2H1,OX1-]"),
    ("sulfonic", "sulfonic acid / sulfonate", "[#6][SX4](=O)(=O)[OX2H1,OX1-]"),
    ("sulfate", "sulfate ester", "[OX2;H0][SX4](=O)(=O)[OX2H1,OX1-]"),
    ("sulfonamide", "sulfonamide", "[SX4](=O)(=O)[#7]"),
    ("phosphate", "phosphate / phosphonate", "[PX4](~[OX1])~[OX2,OX1-]"),
    ("alcohol", "alcohol", "[OX2H1][CX4]"),
    ("ether", "ether", "[OX2]([#6;!$([#6]=[O,S,N])])[#6;!$([#6]=[O,S,N])]"),
    ("ester", "ester", "[#6][CX3](=O)[OX2][#6]"),
    ("amide", "amide", "[NX3][CX3]=[OX1]"),
    ("amine", "amine (neutral)", "[NX3;+0;!$(N~[!#6;!#1]);!$(N[#6]=[O,S,N]);!$(N#*);!$(N=*)]"),
    ("ammonium", "ammonium (protonated amine)", "[NX4+;H1,H2,H3;!$(N~[!#6;!#1])]"),
    ("quaternary", "quaternary ammonium", "[NX4+;H0;!$(N~[!#6])]"),
    ("aromatic", "aromatic ring", "a"),
    ("nitro", "nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("thio", "thiol / thioether", "[$([SX2H1][#6]),$([SX2]([#6])[#6]),$([SX1-][#6])]"),
    ("halogen", "Cl, Br or I", "[Cl,Br,I]"),
)
GROUP_LABELS = {k: label for k, label, _s in GROUPS}
#: PFAS by the OECD 2021 definition: a fully fluorinated methyl or methylene carbon, i.e. a saturated carbon with at
#: least two F and no H, Cl, Br or I attached (-CF3, -CF2-)
PFAS_SMARTS = "[CX4;H0;!$(C~[Cl,Br,I]);$(C(F)F)]"
PFAS_RULE = ("PFAS (OECD 2021): at least one fully fluorinated methyl or methylene carbon, i.e. a saturated carbon "
             "with two or more F and no H, Cl, Br or I (-CF3, -CF2-)")
#: a perfluorinated carbon of a chain: saturated, at least one F, no H, Cl, Br or I. The chain length is the size of
#: the largest connected group of them (PFOA, C7F15-COOH: 7; PFOS: 8)
PF_CARBON_SMARTS = "[CX4;H0;!$(C~[Cl,Br,I]);$(C-F)]"
ALKYL_SMARTS = "[CX4;H2,H3]"
SULFONAMIDE_NPLUS_SMARTS = "[NX4+;$(N[SX4](=O)=O)]"              # a proton on a sulfonamide N (RANK-1's tautomer)
PF_SULFONAMIDE_SMARTS = "[#7][SX4](=O)(=O)[CX4](F)F"             # a perfluoroalkyl sulfonamide (pKa ~6.5 rule)

HEAVY_BINS = ((1, 10), (11, 20), (21, 35), (36, 60), (61, 100), (101, 10**6))
ROT_BINS = ((0, 3), (4, 7), (8, 14), (15, 10**6))
CHAIN_BINS = ((1, 3), (4, 7), (8, 12), (13, 10**6))

GFN2 = ("gfn2-xtb", "tblite-gfn2", "xtb-gfn2")
GFN1 = ("gfn1-xtb", "tblite-gfn1", "xtb-gfn1")
#: refinement engines with an implicit solvent model, preferred in this order for a water run
SOLVENT_GFN2 = ("tblite-gfn2", "xtb-gfn2")


def _family(engine: str) -> str:
    """The ranker family of a refinement engine: gfn2, gfn1, aimnet2, mmff, gfnff, pm7, pm6."""
    if engine in GFN2:
        return "gfn2"
    if engine in GFN1:
        return "gfn1"
    return {"aimnet2": "aimnet2", "mmff": "mmff", "gfnff": "gfnff", "mopac-pm7": "pm7",
            "mopac-pm6": "pm6"}.get(engine, engine)


# ======================================================================================= data
@dataclass
class Group:
    """A count with the molecule ids behind it (the first ``max_ids``)."""
    key: str
    label: str
    n: int = 0
    ids: list[str] = field(default_factory=list)


@dataclass
class Profile:
    n: int = 0
    sections: dict[str, list[Group]] = field(default_factory=dict)   # section -> groups, in display order
    stats: dict[str, dict] = field(default_factory=dict)             # heavy, rotatable, atoms: min/median/max
    seconds: float = 0.0

    def group(self, section: str, key: str) -> Group:
        for g in self.sections.get(section, []):
            if g.key == key:
                return g
        return Group(key, key)

    def count(self, section: str, key: str) -> int:
        return self.group(section, key).n


@dataclass
class Finding:
    level: str                    # "problem" | "warning" | "info"
    topic: str                    # "conformer ranking", "elements", "QM level", "solvent", "cost", "input", ...
    key: str                      # stable id of the rule, e.g. "rank.gfn2-pfas-ions"
    message: str
    why: str = ""                 # one or two sentences: the measured result behind it
    n: int = 0                    # molecules it concerns
    ids: list[str] = field(default_factory=list)
    changes: dict = field(default_factory=dict)      # BuildConfig fields to set
    recommendation: str = ""      # what to do, in words (also when there is no setting to change)
    guide: tuple[str, ...] = ()   # methods-guide entries to read
    evidence: str = ""            # where the measured result is


@dataclass
class CostEstimate:
    lane_hours: float = 0.0       # sum over molecules on one 4-thread lane
    hours: float = 0.0            # on this machine's threads (lane-hours spread over its 4-thread lanes)
    threads: int = 1
    per_molecule_s: float = 0.0   # median
    slowest_s: float = 0.0
    slowest_id: str = ""
    by_stage: dict = field(default_factory=dict)     # stage -> lane-hours
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        return (f"about {_duration(self.hours * 3600)} on this machine ({self.threads} thread"
                f"{'s' if self.threads != 1 else ''}; "
                f"{self.lane_hours:,.1f} lane-hours on 4-thread lanes), median {_duration(self.per_molecule_s)} per "
                f"molecule")


@dataclass
class Preset:
    key: str                      # "fast" | "recommended" | "careful"
    name: str
    changes: dict
    rationale: str
    cost: CostEstimate
    needs: list[str] = field(default_factory=list)   # programs this preset needs that are not found here
    config: object = None         # the BuildConfig (cfg with the changes)
    diff: dict = field(default_factory=dict)         # the changes that differ from the reviewed settings


@dataclass
class CoverageRow:
    key: str                      # method_guide.COVERAGE key
    label: str
    kind: str                     # "refinement" | "conformer search" | "QM basis" | "not in QuantaBench yet"
    n_ok: int = 0
    n_elements: int = 0           # molecules with elements it has no parameters for
    n_spin: int = 0               # open-shell molecules it cannot take
    n_spin_ignored: int = 0       # open-shell molecules it takes without modelling the spin
    ids_excluded: list[str] = field(default_factory=list)
    installed: bool | None = None  # None: not a QuantaBench engine
    note: str = ""


@dataclass
class Review:
    n: int
    settings: dict
    profile: Profile
    coverage: list[CoverageRow]
    findings: list[Finding]
    presets: list[Preset]
    recommended: str
    cost: CostEstimate
    steps: list[tuple[str, str]]
    methods: dict                 # refinement engine -> molecules (what the run would use)
    seconds: float = 0.0

    def counts(self) -> dict[str, int]:
        c = Counter(f.level for f in self.findings)
        return {lv: c.get(lv, 0) for lv in LEVELS}

    def preset(self, key: str) -> Preset | None:
        return next((p for p in self.presets if p.key == key), None)

    def to_dict(self) -> dict:
        d = {"n": self.n, "settings": self.settings, "seconds": round(self.seconds, 3),
             "profile": {"n": self.profile.n, "stats": self.profile.stats,
                         "sections": {s: [asdict(g) for g in gs] for s, gs in self.profile.sections.items()}},
             "coverage": [asdict(r) for r in self.coverage],
             "findings": [asdict(f) for f in self.findings],
             "presets": [{**{k: v for k, v in asdict(p).items() if k != "config"}} for p in self.presets],
             "recommended": self.recommended, "cost": asdict(self.cost), "steps": [list(s) for s in self.steps],
             "methods": self.methods, "counts": self.counts()}
        return _jsonable(d)

    def text(self, width: int = 100, max_ids: int = 8) -> str:
        return render_text(self, width=width, max_ids=max_ids)

    def header(self, max_findings: int = 6) -> str:
        return render_header(self, max_findings=max_findings)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ======================================================================================= per molecule
class _Mol:
    """What the review needs to know about one molecule (the species the run will compute)."""

    __slots__ = ("ok", "error", "z", "charge", "pos", "neg", "mult", "heavy", "atoms", "rot", "tags", "pf_chain",
                 "alkyl", "undef_atoms", "undef_bonds", "n_frags", "removed", "nbf", "radicals", "mmff", "drawn_tags")

    def __init__(self):
        self.ok, self.error = True, ""
        self.z: frozenset[int] = frozenset()
        self.charge = self.pos = self.neg = 0
        self.mult = 1
        self.heavy = self.atoms = self.rot = 0
        self.tags: frozenset[str] = frozenset()
        self.pf_chain = self.alkyl = 0
        self.undef_atoms = self.undef_bonds = 0
        self.n_frags = 1
        self.removed: tuple[str, ...] = ()
        self.nbf = 0
        self.radicals = 0
        self.mmff: bool | None = None        # MMFF94 has parameters (checked only for unusual elements)
        self.drawn_tags: frozenset[str] = frozenset()

    @property
    def charge_class(self) -> str:
        if self.charge < 0:
            return "anion"
        if self.charge > 0:
            return "cation"
        return "zwitterion" if self.pos and self.neg else "neutral"

    @property
    def ion(self) -> bool:
        return self.charge_class != "neutral"

    @property
    def pfas(self) -> bool:
        return "pfas" in self.tags

    @property
    def hydrocarbon(self) -> bool:
        return self.z <= {1, 6} and self.charge_class == "neutral"


_PATTERNS: dict[str, Chem.Mol] = {}


def _patt(smarts: str) -> Chem.Mol:
    p = _PATTERNS.get(smarts)
    if p is None:
        p = _PATTERNS[smarts] = Chem.MolFromSmarts(smarts)
    return p


def _largest_component(mol: Chem.Mol, atoms: set[int]) -> int:
    """Size of the largest connected group of ``atoms`` (bonds between them only)."""
    best, seen = 0, set()
    for start in atoms:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            i = stack.pop()
            size += 1
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in atoms and j not in seen:
                    seen.add(j)
                    stack.append(j)
        best = max(best, size)
    return best


def _nbf_def2svp(z: int) -> int:
    """def2-SVP basis functions of one atom, rounded (PFOA: 355, as the methods guide counts)."""
    if z <= 2:
        return 5
    if z <= 4:
        return 9
    if z <= 10:
        return 14
    if z <= 18:
        return 18
    return 30 if z <= 36 else 25


_STRICT_ROT = rdMolDescriptors.NumRotatableBondsOptions.Strict
_FORMULA = re.compile(r"([A-Z][a-z]?)(\d*)")
_Z_OF = {sym: z for z, sym in enumerate(guide.SYMBOLS) if z}
CHARGED_SMARTS = "[!+0]"


def _formula_counts(mol: Chem.Mol) -> dict[int, int]:
    """Element counts (implicit hydrogens included) from RDKit's formula: a C++ call, about 50x faster than visiting
    the atoms from Python."""
    out: dict[int, int] = {}
    for sym, n in _FORMULA.findall(rdMolDescriptors.CalcMolFormula(mol)):
        z = _Z_OF.get(sym)
        if z is None:
            raise ValueError(f"unknown element {sym!r}")
        out[z] = out.get(z, 0) + (int(n) if n else 1)
    return out


def _drawn_scan(mol: Chem.Mol) -> tuple[list[int], int, bool, bool]:
    """(atomic numbers, unpaired electrons, any isotope label, any query/dummy atom) of the drawn structure."""
    zs, radicals, isotope, query = [], 0, False, False
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        z = a.GetAtomicNum()
        zs.append(z)
        radicals += a.GetNumRadicalElectrons()
        isotope = isotope or bool(a.GetIsotope())
        query = query or z == 0 or a.HasQuery()
    return zs, radicals, isotope, query


def _features(inp, ph: float, protonate: bool, method: str, keep_largest: bool) -> _Mol:
    m = _Mol()
    try:
        with rdBase.BlockLogs():
            drawn = inp.mol if inp.mol is not None else Chem.MolFromSmiles(inp.smiles or "")
        if drawn is None or drawn.GetNumAtoms() == 0:
            raise ValueError(f"could not read the structure ({inp.smiles!r})" if inp.mol is None else "empty structure")
        zs, m.radicals, isotopes, query = _drawn_scan(drawn)
        m.n_frags = len(Chem.GetMolFrags(drawn))
        charged = drawn.HasSubstructMatch(_patt(CHARGED_SMARTS))
        metal = any(z in standardize.METALS for z in zs)
        fast = m.n_frags == 1 and not charged and not m.radicals and not metal and not query and method == "rules"
        with rdBase.BlockLogs():
            if fast:   # what prepare computes for a neutral, single-fragment, metal-free structure: the rules alone
                sp = Chem.RemoveHs(drawn) if 1 in zs else drawn
                if protonate and any(sp.HasSubstructMatch(standardize._rule_pattern(r.smarts)[0])
                                     for r in standardize.RULES):
                    sp, _notes = standardize.protonate(sp, ph, "rules")
            else:
                if drawn.HasSubstructMatch(_patt(SULFONAMIDE_NPLUS_SMARTS)):
                    m.drawn_tags = frozenset({"sulfonamide_nplus"})
                info: dict = {}
                prepared, _notes = standardize.prepare(Chem.Mol(drawn), ph if protonate else None, keep_largest,
                                                       method, info=info)
                sp = Chem.RemoveHs(prepared)
                m.removed = tuple(info.get("removed_fragments") or ())
        formal = Chem.GetFormalCharge(sp)
        m.charge = int(inp.charge) if inp.charge is not None else formal
        if formal or sp.HasSubstructMatch(_patt(CHARGED_SMARTS)):
            for (i,) in sp.GetSubstructMatches(_patt(CHARGED_SMARTS)):
                a = sp.GetAtomWithIdx(i)
                q = a.GetFormalCharge()
                if not any(nb.GetFormalCharge() * q < 0 for nb in a.GetNeighbors()):   # not a nitro-type pair
                    if q > 0:
                        m.pos += 1
                    else:
                        m.neg += 1
        counts = _formula_counts(sp)
        m.z = frozenset(counts)
        m.atoms = sum(counts.values())
        m.heavy = m.atoms - counts.get(1, 0)
        m.nbf = sum(_nbf_def2svp(z) * n for z, n in counts.items())
        requested = inp.multiplicity if inp.multiplicity and inp.multiplicity != 1 else None
        n_el = sum(z * n for z, n in counts.items()) - m.charge
        if requested is not None or n_el % 2 or not fast:
            m.mult, _note = standardize.spin_multiplicity(sp, m.charge, requested)
        m.rot = rdMolDescriptors.CalcNumRotatableBonds(sp, _STRICT_ROT)
        tags = {k for k, _label, smarts in GROUPS if sp.HasSubstructMatch(_patt(smarts))}
        if sp.HasSubstructMatch(_patt(PFAS_SMARTS)):
            tags.add("pfas")
            pf = {t[0] for t in sp.GetSubstructMatches(_patt(PF_CARBON_SMARTS))}
            m.pf_chain = _largest_component(sp, pf)
            if sp.HasSubstructMatch(_patt(PF_SULFONAMIDE_SMARTS)):
                tags.add("pf_sulfonamide")
        if m.charge or m.pos:          # the alkyl chain length is only used for ions (the PCM rule)
            alkyl = {t[0] for t in sp.GetSubstructMatches(_patt(ALKYL_SMARTS))}
            m.alkyl = _largest_component(sp, alkyl) if alkyl else 0
        if sp.HasSubstructMatch(_patt(SULFONAMIDE_NPLUS_SMARTS)):
            tags.add("sulfonamide_nplus")
        if any(len(r) >= 12 for r in sp.GetRingInfo().AtomRings()):
            tags.add("macrocycle")
        if isotopes:
            tags.add("isotope")
        if m.radicals:
            tags.add("radical")
        for si in Chem.FindPotentialStereo(sp):
            if si.specified == Chem.StereoSpecified.Unspecified:
                if si.type == Chem.StereoType.Bond_Double:
                    m.undef_bonds += 1
                elif si.type == Chem.StereoType.Atom_Tetrahedral:
                    m.undef_atoms += 1
        if not m.z <= guide.Z_MMFF94:          # unusual elements: does MMFF94 type this molecule? (rare, so exact)
            from rdkit.Chem import rdForceFieldHelpers

            m.mmff = bool(rdForceFieldHelpers.MMFFHasAllMoleculeParams(Chem.AddHs(sp)))
        m.tags = frozenset(tags)
    except Exception as exc:  # noqa: BLE001 - the reason, for the "cannot be built" finding
        m.ok, m.error = False, f"{type(exc).__name__}: {exc}"
    return m


_CACHE: dict[tuple, _Mol] = {}
_CACHE_MAX = 200_000


def _key(inp, opts: tuple) -> tuple:
    if inp.mol is not None:
        with rdBase.BlockLogs():
            text = "mol:" + Chem.MolToSmiles(inp.mol)
    else:
        text = inp.smiles or ""
    return (text, inp.charge, inp.multiplicity, *opts)


def _options(cfg) -> tuple:
    return (float(cfg.ph), bool(cfg.protonate), str(cfg.protonation_method or "rules"), bool(cfg.keep_largest_fragment))


def molecule_features(inputs, cfg, progress: Callable[[int, int], None] | None = None,
                      should_stop: Callable[[], bool] | None = None) -> list[tuple[str, _Mol]]:
    """(id, features) of every input, from the cache when the same structure was profiled with the same settings."""
    opts = _options(cfg)
    ph, protonate, method, keep = opts
    out = []
    n = len(inputs)
    for i, inp in enumerate(inputs):
        if should_stop is not None and i % 500 == 0 and should_stop():
            break
        key = _key(inp, opts)
        m = _CACHE.get(key)
        if m is None:
            m = _features(inp, ph, protonate, method, keep)
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.clear()
            _CACHE[key] = m
        out.append((inp.id, m))
        if progress is not None and i % 1000 == 999:
            progress(i + 1, n)
    return out


def clear_cache() -> None:
    _CACHE.clear()


# ======================================================================================= profile
class _Groups:
    def __init__(self, max_ids: int):
        self.max_ids = max_ids
        self.sections: dict[str, dict[str, Group]] = {}

    def add(self, section: str, key: str, label: str, ident: str) -> None:
        s = self.sections.setdefault(section, {})
        g = s.get(key)
        if g is None:
            g = s[key] = Group(key, label)
        g.n += 1
        if len(g.ids) < self.max_ids:
            g.ids.append(ident)


def _bin_label(lo: int, hi: int, unit: str = "") -> str:
    return f"{lo}+{unit}" if hi >= 10**6 else f"{lo}-{hi}{unit}" if lo != hi else f"{lo}{unit}"


def _stats(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


SECTION_LABELS = {
    "elements": "Elements", "charge": "Charge class", "net_charge": "Net charge", "pfas": "PFAS",
    "chain": "Perfluoroalkyl chain", "groups": "Functional groups", "heavy": "Heavy atoms",
    "rotatable": "Rotatable bonds", "flags": "Other", "removed": "Fragments the run removes",
}
FLAG_LABELS = {
    "hydrocarbon": "pure hydrocarbons (C, H)", "open_shell": "open shell (radicals, odd electron count)",
    "macrocycle": "macrocycles (a ring of 12 or more atoms)", "undefined_stereo": "undefined stereocentres",
    "undefined_double": "undefined double-bond stereo", "multi_fragment": "several fragments (salts, mixtures)",
    "isotope": "isotope labels", "metal": "metals", "unusual": "other unusual elements",
    "not_computable": "cannot be built", "pf_sulfonamide": "perfluoroalkyl sulfonamides",
    "sulfonamide_nplus": "a proton on a sulfonamide N",
}


def build_profile(mols: list[tuple[str, _Mol]], cfg, max_ids: int = MAX_IDS) -> Profile:
    t0 = time.perf_counter()
    g = _Groups(max_ids)
    heavy, rot, natoms = [], [], []
    for ident, m in mols:
        if not m.ok:
            g.add("flags", "not_computable", FLAG_LABELS["not_computable"], ident)
            continue
        for z in sorted(m.z):
            g.add("elements", guide.SYMBOLS[z], guide.SYMBOLS[z], ident)
        cls = m.charge_class
        g.add("charge", cls, cls, ident)
        g.add("net_charge", f"{m.charge:+d}" if m.charge else "0", f"{m.charge:+d}" if m.charge else "0", ident)
        if m.pfas:
            g.add("pfas", "pfas", "PFAS", ident)
            for lo, hi in CHAIN_BINS:
                if lo <= m.pf_chain <= hi:
                    g.add("chain", _bin_label(lo, hi), f"C{_bin_label(lo, hi)}", ident)
        else:
            g.add("pfas", "not_pfas", "not PFAS", ident)
        for key, label, _s in GROUPS:
            if key in m.tags:
                g.add("groups", key, label, ident)
        for lo, hi in HEAVY_BINS:
            if lo <= m.heavy <= hi:
                g.add("heavy", _bin_label(lo, hi), _bin_label(lo, hi), ident)
        for lo, hi in ROT_BINS:
            if lo <= m.rot <= hi:
                g.add("rotatable", _bin_label(lo, hi), _bin_label(lo, hi), ident)
        heavy.append(m.heavy)
        rot.append(m.rot)
        natoms.append(m.atoms)
        if m.hydrocarbon:
            g.add("flags", "hydrocarbon", FLAG_LABELS["hydrocarbon"], ident)
        if m.mult > 1:
            g.add("flags", "open_shell", FLAG_LABELS["open_shell"], ident)
        for tag in ("macrocycle", "isotope", "pf_sulfonamide", "sulfonamide_nplus"):
            if tag in m.tags:
                g.add("flags", tag, FLAG_LABELS[tag], ident)
        if m.undef_atoms:
            g.add("flags", "undefined_stereo", FLAG_LABELS["undefined_stereo"], ident)
        if m.undef_bonds:
            g.add("flags", "undefined_double", FLAG_LABELS["undefined_double"], ident)
        if m.n_frags > 1:
            g.add("flags", "multi_fragment", FLAG_LABELS["multi_fragment"], ident)
        for frag in m.removed:
            g.add("removed", frag, frag, ident)
        metals = [z for z in m.z if z in standardize.METALS]
        if metals:
            g.add("flags", "metal", FLAG_LABELS["metal"], ident)
        if any(z not in COMMON_ELEMENTS and z not in standardize.METALS for z in m.z):
            g.add("flags", "unusual", FLAG_LABELS["unusual"], ident)
    order = {"charge": ("neutral", "anion", "cation", "zwitterion"), "pfas": ("pfas", "not_pfas")}
    sections: dict[str, list[Group]] = {}
    for section in ("elements", "charge", "net_charge", "pfas", "chain", "groups", "heavy", "rotatable", "flags",
                    "removed"):
        groups = list(g.sections.get(section, {}).values())
        if section == "elements":
            groups.sort(key=lambda x: (x.key not in ("C", "H"), -x.n, x.key))
        elif section in order:
            groups.sort(key=lambda x: order[section].index(x.key) if x.key in order[section] else 99)
        elif section == "net_charge":
            groups.sort(key=lambda x: int(x.key))
        elif section == "groups":
            keys = [k for k, _l, _s in GROUPS]
            groups.sort(key=lambda x: keys.index(x.key))
        elif section in ("heavy", "rotatable", "chain"):
            groups.sort(key=lambda x: int(x.key.split("-")[0].rstrip("+")))
        elif section == "flags":
            keys = list(FLAG_LABELS)
            groups.sort(key=lambda x: keys.index(x.key))
        else:
            groups.sort(key=lambda x: -x.n)
        sections[section] = groups
    return Profile(n=len(mols), sections=sections,
                   stats={"heavy": _stats(heavy), "rotatable": _stats(rot), "atoms": _stats(natoms)},
                   seconds=time.perf_counter() - t0)


def profile_batch(inputs, cfg, max_ids: int = MAX_IDS) -> tuple[Profile, list[tuple[str, _Mol]]]:
    mols = molecule_features(list(inputs), cfg)
    return build_profile(mols, cfg, max_ids), mols


# ======================================================================================= availability
def probe_available() -> set[str]:
    """Engine keys installed here (quick checks: no torch import), plus "psi4" and "crest" when found."""
    from . import engines

    out = {k for k in engines.all_engines() if engines.available(k, quick=True)[0]}
    try:
        from . import qm

        if qm.available()[0]:
            out.add("psi4")
    except Exception:  # noqa: BLE001, S110 - a broken Psi4 setup: not available
        pass
    try:
        from . import crest

        if crest.available()[0]:
            out.add("crest")
    except Exception:  # noqa: BLE001, S110 - a broken CREST setup: not available
        pass
    return out


class _Resolver:
    """Which refinement engine the pipeline will use for a molecule (``pipeline.run_molecule``'s choice, from the
    installed engines ``available``), cached by elements."""

    def __init__(self, cfg, solvent: str, available: set[str]):
        from . import engines

        self.engines = engines
        self.cfg, self.solvent, self.available = cfg, solvent, available
        self._cache: dict[frozenset, tuple[str, str]] = {}

    def auto(self, z: frozenset) -> str:
        e = self.engines
        keys = [k for k in e.AUTO_ORDER if k in self.available and k in e.all_engines() and e.supports(k, z)[0]]
        if self.solvent:
            keys = ([k for k in keys if e.supports_solvent(k, self.solvent)]
                    + [k for k in keys if not e.supports_solvent(k, self.solvent)])
        return keys[0] if keys else "mmff"

    def __call__(self, z: frozenset) -> tuple[str, str]:
        """(engine, why it is not the configured one: '' when it is)."""
        hit = self._cache.get(z)
        if hit is not None:
            return hit
        method = self.cfg.method
        if method == "auto":
            out = (self.auto(z), "")
        elif method not in self.engines.all_engines():
            out = (self.auto(z), f"{method} is not an engine")
        elif method != "mmff" and method not in self.available:
            out = (self.auto(z), f"{method} is not installed here")
        else:
            ok, why = self.engines.supports(method, z)
            out = (method, "") if ok else (self.auto(z), why)
        self._cache[z] = out
        return out


# ======================================================================================= the measured ranking table
@dataclass(frozen=True)
class Verdict:
    level: str
    message: str
    why: str
    evidence: str


def rank_verdict(family: str, cls: str, phase: str) -> Verdict | None:
    """How well a refinement method family ranks conformers of a molecule class in a phase, as measured.

    ``family``: gfn2, gfn1, aimnet2, mmff, gfnff, pm7, pm6. ``cls``: "pfas_ion", "ion" (other ions and
    zwitterions), "polar" (neutral, not a pure hydrocarbon), "hydrocarbon". ``phase``: "gas" or "water"
    (any solvent). None when there is nothing measured to say."""
    water = phase != "gas"
    if cls == "pfas_ion":
        if family == "gfn2":
            return Verdict("problem", "GFN2-xTB ranks the conformers of PFAS ions no better than a random pick",
                           "Against r2SCAN-3c in PCM water its lowest conformer was within 4 kJ/mol of the reference's "
                           "in 0 of 13 PFAS ions (regret 36 kJ/mol, 1.6 times a random pick's), and against "
                           "ωB97X-D/def2-SVP in 4 of 17, in the gas phase, ALPB or CPCM alike; it overbinds short "
                           "anionic O···F contacts by 15-20 kJ/mol each.", f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}")
        if family in ("pm7", "mmff", "gfnff"):
            name = {"pm7": "PM7", "mmff": "MMFF94s", "gfnff": "GFN-FF"}[family]
            why = {"pm7": "PM7 picked within 4 kJ/mol of ωB97X-D/PCM's lowest in 5 of 17 PFAS ions (COSMO; median ρ "
                          "−0.05) and overbinds O···F contacts as GFN2-xTB does.",
                   "mmff": "As single points its relative energies were 51 kJ/mol off ωB97X-D/PCM's for PFAS ions "
                           "(7 of 17 picks within 4 kJ/mol).",
                   "gfnff": "GFN-FF's errors for PFAS anions reached 100 kJ/mol; it is a search level, not a "
                            "ranking."}[family]
            return Verdict("problem", f"{name} does not rank the conformers of PFAS ions", why, EVIDENCE_RANK1)
        if family == "aimnet2":
            return Verdict("warning", "AIMNet2 ranks PFAS ions only moderately, and in the gas phase",
                           "It picked within 4 kJ/mol of the DFT lowest in 12 of 17 PFAS ions against "
                           "ωB97X-D/def2-SVP (regret up to 26 kJ/mol) and 10 of 13 against r2SCAN-3c, with no "
                           "solvent model; a CPCM solvation term (the coming rank step) improves it.",
                           f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}")
        if family in ("gfn1", "pm6"):
            name = "GFN1-xTB" if family == "gfn1" else "PM6-D3H4X"
            why = ("GFN1-xTB with tblite's CPCM picked within 4 kJ/mol of ωB97X-D/PCM's lowest in 11 of 17 PFAS ions; "
                   "the pipeline's ALPB was not measured." if family == "gfn1" else
                   "With COSMO it picked within 4 kJ/mol in 13 of 17 PFAS ions but missed by up to 35 kJ/mol; in the "
                   "gas phase it overbinds O···F contacts like GFN2-xTB.")
            return Verdict("warning", f"{name}'s ranking of PFAS ions is only partly reliable", why, EVIDENCE_RANK1)
        return None
    if cls == "ion":
        if family in ("gfn2", "aimnet2", "gfn1") and water:
            name = {"gfn2": "GFN2-xTB", "aimnet2": "AIMNet2", "gfn1": "GFN1-xTB"}[family]
            return Verdict("warning", f"{name} ranks the conformers of these ions about as well as a random pick",
                           "On 8 non-fluorinated ions in water (carboxylates, ammoniums) GFN2-xTB/ALPB picked within "
                           "4 kJ/mol of r2SCAN-3c/PCM in 4 (regret 4.5 kJ/mol, 0.73 of a random pick's); no xTB or "
                           "AIMNet2 variant ranked the carboxylates well. Their pools are flat, so the misses are a "
                           "few kJ/mol, not tens.", EVIDENCE_RANK1B)
        return None
    if cls == "polar":
        if family == "gfn2" and water:
            return Verdict("warning", "GFN2-xTB/ALPB ranks the conformers of polar molecules in water poorly",
                           "In water it picked within 4 kJ/mol of r2SCAN-3c/PCM in 16 of 28 polar organics and 14 of "
                           "20 neutral PFAS (median ρ 0.31-0.36, 0.6-0.7 of a random pick's regret); g-xTB + a "
                           "GFN1-xTB CPCM term did so in 26 of 28 and 19 of 20.", EVIDENCE_RANK1B)
        if family == "gfn2":
            return Verdict("info", "GFN2-xTB ranks gas-phase conformers of polar molecules less well than AIMNet2",
                           "In the gas phase GFN2-xTB picked within 4 kJ/mol of r2SCAN-3c in 22 of 28 polar organics "
                           "and 15 of 20 PFAS, AIMNet2 in 27 of 28 and 18 of 20 (Essential 76 against ωB97X-D/6-31G*: "
                           "36 and 45 of 53).", f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}")
        if family == "aimnet2" and not water:
            return Verdict("info", "AIMNet2 is the best cheap gas-phase ranker measured for neutral molecules",
                           "It picked within 4 kJ/mol of r2SCAN-3c in 39 of 40 non-fluorinated neutrals and 18 of 20 "
                           "PFAS; their dipoles hardly depend on the conformer picked.",
                           f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}")
        if family == "aimnet2":
            return Verdict("warning", "AIMNet2 has no solvent model: this water run ranks in the gas phase",
                           "As a gas-phase ranker of water pools it did well (25 of 28 polar organics, 19 of 20 PFAS "
                           "within 4 kJ/mol), but the run's geometries and properties are then gas-phase ones.",
                           EVIDENCE_RANK1B)
        if family in ("mmff", "gfnff", "pm7"):
            name = {"mmff": "MMFF94s", "gfnff": "GFN-FF", "pm7": "PM7"}[family]
            nums = {"mmff": "19 of 28 polar organics and 10 of 20 PFAS in the gas phase (Essential 76: 30 of 53)",
                    "gfnff": "22 of 53 Essential 76 PFAS against ωB97X-D/6-31G*, no better than a random pick",
                    "pm7": "20 of 28 polar organics in the gas phase (Essential 76: 35 of 53)"}[family]
            return Verdict("warning", f"{name} ranks the conformers of polar molecules poorly",
                           f"Its pick was within 4 kJ/mol of the DFT lowest in {nums}.",
                           f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}")
        return None
    if cls == "hydrocarbon" and family in ("gfn2", "gfn1", "aimnet2", "mmff", "pm7", "pm6"):
        name = {"gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB", "aimnet2": "AIMNet2", "mmff": "MMFF94s", "pm7": "PM7",
                "pm6": "PM6-D3H4X"}[family]
        return Verdict("info", f"{name} ranks pure hydrocarbons well",
                       "Every cheap method tested picked within 4 kJ/mol of r2SCAN-3c in 10-12 of 12 hydrocarbons "
                       "(GFN2-xTB 12 of 12, gas and water): their pools span little.", EVIDENCE_RANK1B)
    return None


# ======================================================================================= cost model
#: Measured anchors (QuantaBench, docs/VALIDATION.md, ENGINES.md, PLATFORMS.md, BACKLOG PERF-1); everything else is
#: scaled from them by size, so an estimate is good to a factor of about 2.
DFT_SP_S = 300.0          # ωB97X-D/def2-SVP IEF-PCM single point, 495 basis functions, one 4-thread NOVA lane
DFT_NBF0 = 495.0
DFT_POWER = 2.9           # time grows as nbf^2.9 (the DFT conformer test)
BASIS_FACTOR = {"def2-svp": 1.0, "def2-svpd": 3.0, "def2-tzvp": 7.0, "6-31g*": 1.25, "6-31+g*": 2.0,
                "6-311+g**": 3.0, "aug-cc-pvdz": 3.5, "def2-tzvpd": 10.0, "def2-tzvpp": 8.0}   # the last two assumed
#: ωB97X-D = 1; measured: B3LYP-D3(BJ) 0.72 (DFT test), r2SCAN-3c 2.0, HF-3c 0.25 (RANK-1); the rest assumed
FUNCTIONAL_FACTOR = {"wb97x-d": 1.0, "b3lyp-d3bj": 0.72, "b3lyp": 0.72, "pbe0-d3bj": 0.75, "pbe0": 0.75,
                     "r2scan": 0.6, "r2scan-d4": 0.6, "m06-2x": 1.2, "hf": 0.7, "wb97m-v": 1.3}
QM_PROPERTIES_FACTOR = 0.7     # the best conformer's properties and finite-field polarizability: +70 % (857 vs 507 s)
CREST_GFN2_QUICK_S = 120.0     # PFBA (14 atoms) with CREST --gfn2 --quick on 2 threads: about 2 min (chem/crest.py)
CREST_ATOMS0 = 14.0
CREST_LEVEL_FACTOR = {"gfn2": 1.0, "gfn1": 1.0, "gfnff": 1 / 6, "gff": 1 / 6, "gfn2//gfnff": 0.5}
CREST_FULL_FACTOR = 4.0        # a full search instead of --quick: not measured in QuantaBench (a guess)
#: one optimization step of a 25-atom molecule on one core, seconds; about 60 steps per conformer
STEP_S = {"gfn2": 0.015, "gfn1": 0.012, "aimnet2": 0.12, "gfnff": 0.002, "pm7": 0.02, "pm6": 0.02, "mmff": 0.0}
OPT_STEPS = 60
CONF_EMBED_S = 0.028           # ETKDGv3 + MMFF94s per conformer for a 25-atom PFAS (24 PFAS, 50 conformers: 34 s)
LANE_THREADS = 4


def _qm_seconds(m: _Mol, level: str) -> float:
    from . import qm

    try:
        func, basis = qm.parse_level(level)
    except ValueError:
        return 0.0
    b = BASIS_FACTOR.get(basis, 2.0)
    f = FUNCTIONAL_FACTOR.get(func, 1.0)
    return DFT_SP_S * (max(m.nbf, 50) / DFT_NBF0) ** DFT_POWER * b * f


def _crest_seconds(m: _Mol, cfg, model) -> float:
    level = str(cfg.crest_method or "gfn2").lower()
    if model is not None and getattr(model, "scale_s", None):
        s = model.seconds(m.atoms, LANE_THREADS)
        if s is not None:
            return float(s) * (1.0 if cfg.crest_quick else CREST_FULL_FACTOR)
    base = CREST_GFN2_QUICK_S * (max(m.atoms, 4) / CREST_ATOMS0) ** 4.0 * (LANE_THREADS / 2) ** -0.6
    return base * CREST_LEVEL_FACTOR.get(level, 1.0) * (1.0 if cfg.crest_quick else CREST_FULL_FACTOR)


def molecule_seconds(m: _Mol, cfg, engine: str, crest_model=None) -> dict[str, float]:
    """Estimated seconds of each stage of one molecule on a 4-thread lane."""
    size = max(m.atoms, 4) / 25.0
    out = {}
    if cfg.conformer_search == "crest":
        out["conformers"] = _crest_seconds(m, cfg, crest_model)
        n_conf = max(1, min(cfg.refine_top, 30))
    else:
        out["conformers"] = cfg.max_conformers * CONF_EMBED_S * size ** 1.9
        n_conf = max(1, min(cfg.refine_top, cfg.max_conformers, 10 + m.rot * 3))
    fam = _family(engine)
    step = STEP_S.get(fam, 0.015) * size ** (1.5 if fam == "aimnet2" else 2.5)
    out["refine"] = n_conf * OPT_STEPS * step
    if cfg.xtb_properties:
        out["properties"] = 40 * STEP_S["gfn1"] * size ** 2.5
    if cfg.frequencies:
        out["frequencies"] = 6 * max(m.atoms, 2) * STEP_S.get(fam, 0.015) * size ** 2.5
    if cfg.qm_method:
        sp = _qm_seconds(m, cfg.qm_method)
        n_qm = max(1, min(cfg.qm_top, n_conf))
        out["qm"] = sp * (n_qm + (QM_PROPERTIES_FACTOR if cfg.qm_properties else 0.0)) \
            * (3.0 if cfg.qm_task == "optimize" else 1.0)
        if cfg.qm_frequencies:
            out["qm"] += sp * 6 * max(m.atoms, 2) * 0.5
    return out


def estimate_cost(mols: list[tuple[str, _Mol]], cfg, resolve: _Resolver) -> CostEstimate:
    from .. import resources

    model, notes = None, []
    if cfg.conformer_search == "crest":
        try:
            from ..dist import threads as dthreads

            model = dthreads.cost_model_for(cfg)
            if model is not None and model.scale_s:
                notes.append(f"CREST: this machine's fitted cost model ({model.source})")
            else:
                notes.append("CREST: built-in model (atoms^4, anchored at PFBA ~2 min on 2 threads with --gfn2 "
                             "--quick); `quantabench-cli timings --fit` fits this machine's")
        except Exception:  # noqa: BLE001
            model = None
    if cfg.qm_method:
        notes.append("QM: ωB97X-D/def2-SVP in PCM at 5 min for 495 basis functions on a 4-thread NOVA lane, "
                     "growing as nbf^2.9 (the DFT conformer test); other levels scaled by their measured cost")
    per, by_stage = [], Counter()
    slowest, slowest_id = 0.0, ""
    for ident, m in mols:
        if not m.ok:
            continue
        engine, _why = resolve(m.z)
        stages = molecule_seconds(m, cfg, engine, model)
        total = sum(stages.values())
        per.append(total)
        by_stage.update(stages)
        if total > slowest:
            slowest, slowest_id = total, ident
    try:
        threads = resources.resolve_threads(int(cfg.threads or 0))
    except Exception:  # noqa: BLE001
        threads = 1
    lane_h = sum(per) / 3600.0
    # CPU-bound work spread over the machine's threads in 4-thread lanes; one molecule cannot use more than its lane
    lanes = threads / LANE_THREADS
    lanes = min(lanes, max(1, len(per))) if lanes >= 1 else lanes
    hours = lane_h / lanes if per else 0.0
    slowest_h = slowest / 3600.0 / min(1.0, threads / LANE_THREADS)
    return CostEstimate(lane_hours=lane_h, hours=max(hours, slowest_h), threads=threads,
                        per_molecule_s=statistics.median(per) if per else 0.0, slowest_s=slowest,
                        slowest_id=slowest_id, by_stage={k: v / 3600.0 for k, v in by_stage.items()}, notes=notes)


def _duration(seconds: float) -> str:
    s = float(seconds or 0.0)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 3 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


# ======================================================================================= the review
class _Ctx:
    def __init__(self, mols, cfg, available, max_ids):
        from . import solvents

        self.cfg = cfg
        self.solvent = solvents.canonical(cfg.solvent)
        self.phase = "water" if self.solvent else "gas"
        self.mols = [(i, m) for i, m in mols if m.ok]
        self.by_id = dict(self.mols)
        self.failed = [(i, m) for i, m in mols if not m.ok]
        self.available = available
        self.max_ids = max_ids
        self.resolve = _Resolver(cfg, self.solvent, available)
        self.engine = {i: self.resolve(m.z)[0] for i, m in self.mols}

    def cls(self, m: _Mol) -> str:
        if m.ion:
            return "pfas_ion" if m.pfas else "ion"
        return "hydrocarbon" if m.hydrocarbon else "polar"

    def finding(self, level, topic, key, message, ids, **kw) -> Finding:
        ids = list(ids)
        return Finding(level, topic, key, message, n=len(ids), ids=ids[:self.max_ids], **kw)

    def gfn2_engine(self, solvent: bool) -> str:
        """The installed GFN2-xTB engine for this phase ('' when there is none)."""
        order = SOLVENT_GFN2 if solvent else ("gfn2-xtb", *SOLVENT_GFN2)
        return next((k for k in order if k in self.available), "")


def _qm_parts(level: str) -> tuple[str, str]:
    from . import qm

    try:
        return qm.parse_level(level)
    except ValueError:
        return "", ""


def has_dispersion(functional: str) -> bool:
    """Whether a Psi4 functional name includes a dispersion correction (-D, -D3, -D3BJ, -D4, -V, a -3c composite)."""
    f = (functional or "").lower().replace("(", "").replace(")", "")
    return (any(s in f for s in ("-d3", "-d4", "-d2", "d3bj", "-nl", "3c")) or f.endswith(("-d", "-v"))
            or f in ("wb97x-d", "wb97m-v", "wb97x-v", "b97m-v", "b97-d", "b97-d3"))


def has_diffuse(basis: str) -> bool:
    b = (basis or "").lower()
    return "+" in b or b.startswith(("aug-", "jun-", "may-", "d-aug")) or (b.startswith("def2-") and b.endswith("d"))


#: the functional to use when one without dispersion was chosen
DISPERSION_FIX = {"b3lyp": "b3lyp-d3bj", "pbe0": "pbe0-d3bj", "pbe": "pbe-d3bj", "r2scan": "r2scan-d4",
                  "tpss": "tpss-d3bj", "blyp": "blyp-d3bj"}
#: the diffuse version of a basis set, for anions
DIFFUSE_FIX = {"def2-svp": "def2-svpd", "def2-tzvp": "def2-tzvpd", "def2-tzvpp": "def2-tzvppd",
               "def2-qzvp": "def2-qzvpd", "6-31g*": "6-31+g*", "6-31g**": "6-31+g**", "6-311g**": "6-311+g**",
               "cc-pvdz": "aug-cc-pvdz", "cc-pvtz": "aug-cc-pvtz"}


def _basis_name(basis: str) -> str:
    e = guide.lookup(basis)
    return e.name if e is not None and e.family == "Basis sets" else basis


def corrected_level(cfg, anions: bool) -> str:
    """The QM level the findings recommend, so that applying one never undoes another: the configured functional
    when it has a dispersion correction (else its dispersion-corrected form, or ωB97X-D), and for a batch with anions
    the diffuse form of the configured basis (def2-SVPD when it has none). Without a QM stage: ωB97X-D/def2-SVP(D)."""
    func, basis = _qm_parts(cfg.qm_method) if cfg.qm_method else ("", "")
    if not func or func.replace("-", "").endswith("3c"):
        func, basis = "wb97x-d", basis if func else ""
    elif not has_dispersion(func):
        func = DISPERSION_FIX.get(func, "wb97x-d")
    basis = basis or "def2-svp"
    if anions and not has_diffuse(basis):
        basis = DIFFUSE_FIX.get(basis, "def2-svpd")
    return f"{func}/{basis}"


def _anionic(m: _Mol) -> bool:
    return m.charge < 0 or bool(m.neg)


def rerank_configured(cfg, anions: bool, solvent: str) -> bool:
    """Whether ``cfg`` already re-ranks a wide pool with DFT the way the findings recommend: a dispersion-corrected
    functional, a diffuse basis for anions, at least 8 conformers, a window of at least 15 kcal/mol, in a solvent."""
    func, basis = _qm_parts(cfg.qm_method) if cfg.qm_method else ("", "")
    return bool(func and solvent and has_dispersion(func) and (has_diffuse(basis) or not anions)
                and int(cfg.qm_top or 1) >= 8 and float(cfg.energy_window_kcal) >= 15.0)


def _rank_findings(ctx: _Ctx) -> list[Finding]:
    """Conformer ranking: the measured verdict for each (refinement family, molecule class) in the run's phase."""
    by: dict[tuple[str, str], list[str]] = {}
    for ident, m in ctx.mols:
        by.setdefault((_family(ctx.engine[ident]), ctx.cls(m)), []).append(ident)
    out = []
    cfg = ctx.cfg
    psi4 = "psi4" in ctx.available
    for (fam, cls), ids in sorted(by.items(), key=lambda kv: -len(kv[1])):
        v = rank_verdict(fam, cls, ctx.phase)
        if v is None:
            continue
        changes, rec = {}, ""
        level_of = v.level
        message = v.message
        guide_keys: tuple[str, ...] = ("conformer-ranking", guide.ENGINE_ENTRIES.get(ctx.engine[ids[0]], ""))
        anions = any(_anionic(m) for _i, m in ctx.mols)
        if (v.level != "info" and cls in ("pfas_ion", "ion", "polar") and ctx.phase != "gas"
                and rerank_configured(cfg, anions, ctx.solvent)):
            level_of = "info"
            message = (f"{v.message}; the QM stage re-ranks the lowest {cfg.qm_top} at {cfg.qm_method} in PCM, as "
                       "measured best, and its qm_* descriptors follow the DFT-lowest conformer")
            rec = ("The other descriptors (xtb_*, 3D shape, Boltzmann averages) stay at the refinement's ranking until "
                   "the rank step (BACKLOG RANK-2) lands.")
            guide_keys = ("conformer-ranking", "pcm")
        elif v.level != "info" and cls in ("pfas_ion", "ion", "polar") and (cls != "polar" or ctx.phase != "gas"):
            level = corrected_level(cfg, anions)
            changes = {"qm_method": level, "qm_top": max(int(cfg.qm_top or 1), 8),
                       "energy_window_kcal": max(float(cfg.energy_window_kcal), 15.0),
                       "refine_top": max(int(cfg.refine_top), 20)}
            if not ctx.solvent and cls != "polar":
                changes["solvent"] = "water"
            basis = level.split("/", 1)[1]
            rec = ("Today: re-rank a wider pool with DFT in PCM water: the QM stage on the lowest 8 conformers at "
                   f"{level}" + (f" ({_basis_name(basis)}'s diffuse functions for the anions)" if anions else "")
                   + ", keeping conformers up to 15 kcal/mol (63 kJ/mol) above the refinement's lowest, since DFT's "
                   "lowest sat 5-62 kJ/mol above GFN2's lowest in the DFT test. Only the qm_* descriptors follow the "
                   "DFT-lowest conformer; the coming rank step (BACKLOG RANK-2: g-xTB + GFN1-xTB CPCM, about 0.1 % "
                   "of the DFT cost) makes every descriptor follow it."
                   + ("" if psi4 else " Psi4 is not found on this machine."))
            guide_keys = ("conformer-ranking", guide.ENGINE_ENTRIES.get(ctx.engine[ids[0]], ""), "g-xtb",
                          "def2-svpd" if anions else "pcm")
        elif fam == "gfn2" and cls == "polar" and ctx.phase == "gas" and "aimnet2" in ctx.available:
            covered = [i for i in ids if _aimnet_ok(ctx.by_id[i])]
            every = all(_aimnet_ok(m) for _i, m in ctx.mols)
            open_shell = any(m.mult > 1 for _i, m in ctx.mols)
            if covered and (every or not open_shell):
                changes = {"method": "aimnet2" if every else "auto"}
                rec = ("Refine with AIMNet2" + ("." if every else ": method auto takes it for every molecule within "
                                                "its 14 elements and the next engine for the others."))
            elif covered:
                rec = ("Refine the closed-shell molecules within AIMNet2's 14 elements with AIMNet2, in a batch of "
                       "their own (auto would also send the open-shell ones to it).")
            guide_keys = ("conformer-ranking", "aimnet2")
        out.append(ctx.finding(level_of, "conformer ranking", f"rank.{fam}.{cls}", f"{message} ({len(ids)} "
                               f"molecule{'s' if len(ids) != 1 else ''}).", ids, why=v.why, changes=changes,
                               recommendation=rec, guide=tuple(k for k in guide_keys if k), evidence=v.evidence))
    # the window: GFN2's 6 kcal/mol window drops DFT's lowest before the QM stage sees it
    ions_gfn2 = [i for i, m in ctx.mols if m.pfas and m.ion and _family(ctx.engine[i]) == "gfn2"]
    if ions_gfn2 and cfg.qm_method and cfg.energy_window_kcal < 15:
        out.append(ctx.finding(
            "warning", "conformer ranking", "rank.window",
            f"The {cfg.energy_window_kcal:g} kcal/mol window is pruned at GFN2-xTB energies before the QM stage sees "
            "the pool.", ions_gfn2,
            why="GFN2-xTB's 6 kcal/mol (25 kJ/mol) window kept DFT's lowest conformer in only 11 of 17 PFAS ions.",
            changes={"energy_window_kcal": 15.0}, recommendation="Widen the window to 15 kcal/mol.",
            guide=("conformer-ranking", "boltzmann"), evidence=EVIDENCE_RANK1))
    return out


def _aimnet_ok(m: _Mol) -> bool:
    return guide.COVERAGE["aimnet2"].handles(m.z, m.charge, m.mult)[0]


def _element_findings(ctx: _Ctx) -> list[Finding]:
    cfg = ctx.cfg
    out = []
    # the configured method cannot take some molecules: the pipeline falls back per molecule
    fallback = [(i, ctx.resolve(m.z)) for i, m in ctx.mols]
    moved = [i for i, (_e, why) in fallback if why]
    if moved:
        why0 = next(why for _i, (_e, why) in fallback if why)
        to = Counter(e for _i, (e, why) in fallback if why)
        common = _common_method(ctx)
        targets = ", ".join(f"{k} ({n})" for k, n in to.most_common())
        mixed = len(set(ctx.engine.values())) > 1
        out.append(ctx.finding(
            "problem" if mixed else "warning", "elements", "elements.fallback",
            f"{cfg.method} cannot take {len(moved)} molecule(s) ({why0}): the pipeline refines them with {targets} "
            "instead" + (", so the batch mixes levels." if mixed else "."), moved,
            why=("Energies and dipoles from different methods are not comparable within one table." if mixed else
                 "The run notes the fallback for every molecule."),
            changes={"method": common} if common else {},
            recommendation=(f"Use {common} for the whole batch" + (", or split it." if mixed else ".") if common
                            else "Split the batch."),
            guide=("auto",), evidence="method_guide.COVERAGE"))
    engines_used = Counter(ctx.engine.values())
    if cfg.method == "auto" and len(engines_used) > 1:
        from . import engines as _engines

        common = _common_method(ctx)
        top = ", ".join(f"{k} ({n})" for k, n in engines_used.most_common())
        order = list(_engines.AUTO_ORDER)
        primary = min(engines_used, key=lambda e: (-engines_used[e], order.index(e) if e in order else 99))
        minority = [i for i, e in ctx.engine.items() if e != primary]
        out.append(ctx.finding(
            "warning", "elements", "elements.mixed-auto",
            f"Method auto refines this batch with several engines: {top}.", minority,
            why="auto takes the best installed engine each molecule's elements allow; compare energies and dipoles "
                "only within one method (summary.csv: method).",
            changes={"method": common} if common else {},
            recommendation=f"Set {common} for every molecule if they must be comparable." if common else "",
            guide=("auto",), evidence="method_guide.COVERAGE"))
    # open shells: closed-shell models and force fields
    for ident_list, key, level, msg, why in _spin_problems(ctx):
        gfn2 = ctx.gfn2_engine(bool(ctx.solvent))
        out.append(ctx.finding(level, "elements", key, msg, ident_list, why=why,
                               changes={"method": gfn2} if gfn2 and level == "problem" else {},
                               recommendation=(f"Refine them with {gfn2} (unrestricted)." if gfn2 else
                                               "Use an xTB engine (unrestricted)."),
                               guide=("aimnet2", "gfn2-xtb"), evidence="AIMNet2 model card; method_guide.COVERAGE"))
    beyond = [i for i, m in ctx.mols if any(z > 86 for z in m.z)]
    if beyond:
        out.append(ctx.finding(
            "problem", "elements", "elements.beyond-rn",
            f"{len(beyond)} molecule(s) contain elements beyond Rn: no quantum method in QuantaBench covers them.",
            beyond, why="GFN1/GFN2-xTB and GFN-FF stop at Rn (Z = 86); only UFF covers them, and g-xTB (Z 1-103) is "
                        "not an engine yet.", recommendation="Leave them out, or build them with method mmff (UFF).",
            guide=("gfn2-xtb", "g-xtb", "uff"), evidence="method_guide.COVERAGE"))
    unusual = [i for i, m in ctx.mols if any(z not in COMMON_ELEMENTS for z in m.z)]
    if unusual:
        els = sorted({z for i in unusual for z in ctx.by_id[i].z if z not in COMMON_ELEMENTS})
        covers = [c.label for c in guide.COVERAGE.values()
                  if not c.key.startswith("crest-") and all(not c.missing(ctx.by_id[i].z) for i in unusual)
                  and (c.key not in ("g-xtb",))]
        out.append(ctx.finding(
            "info", "elements", "elements.unusual",
            f"{len(unusual)} molecule(s) contain {', '.join(guide.SYMBOLS[z] for z in els)}; methods that cover "
            f"all of them: {', '.join(covers) or 'none'}.", unusual,
            why="Element coverage from each method's documentation (quantabench-cli methods, the coverage table); "
                "RANK-1b's polar organics included Si, P and I, so the ranking results apply to them.",
            guide=("auto",), evidence="method_guide.COVERAGE"))
    no_mmff = [i for i, m in ctx.mols if m.mmff is False]
    if no_mmff and cfg.conformer_search != "crest":
        els = sorted({guide.SYMBOLS[z] for i, m in ctx.mols if m.mmff is False for z in m.z
                      if z not in guide.Z_MMFF94})
        out.append(ctx.finding(
            "info", "elements", "elements.uff",
            f"MMFF94s has no parameters for {len(no_mmff)} molecule(s) ({', '.join(els)}): their conformer pool is "
            "pre-optimized with UFF.", no_mmff,
            why="MMFF94 covers C, H, N, O, F, Si, P, S, Cl, Br, I and a few free ions (Halgren 1996; checked with "
                "RDKit); the refinement then re-optimizes the pool.", guide=("mmff94s", "uff"),
            evidence="method_guide.COVERAGE"))
    if cfg.conformer_search == "crest":
        cov = guide.coverage_for_crest(cfg.crest_method)
        if cov is not None:
            bad = [i for i, m in ctx.mols if not cov.handles(m.z, m.charge, m.mult)[0]]
            if bad:
                out.append(ctx.finding("problem", "elements", "elements.crest",
                                       f"CREST ({cov.label}) cannot take {len(bad)} molecule(s).", bad,
                                       why=cov.handles(ctx.by_id[bad[0]].z)[1], guide=("crest",),
                                       recommendation="They fall back to RDKit's conformers (crest_fallback).",
                                       evidence="method_guide.COVERAGE"))
    if cfg.qm_method:
        func, basis = _qm_parts(cfg.qm_method)
        cov = guide.coverage_for_basis(basis) if basis else None
        if basis and cov is None:
            out.append(ctx.finding("info", "QM level", "qm.basis-unknown",
                                   f"The review does not know which elements {basis} covers.", [],
                                   guide=("basis-sets",)))
        elif cov is not None:
            bad = [i for i, m in ctx.mols if cov.missing(m.z)]
            if bad:
                els = sorted({guide.SYMBOLS[z] for i in bad for z in cov.missing(ctx.by_id[i].z)})
                fix = "def2-svpd" if has_diffuse(basis) else "def2-svp"
                ok = all(not guide.COVERAGE[fix].missing(ctx.by_id[i].z) for i in bad)
                out.append(ctx.finding(
                    "problem", "elements", "qm.basis-elements",
                    f"{cov.label} has no functions for {', '.join(els)} ({len(bad)} molecule(s)): their QM stage "
                    "fails.", bad, why=f"Psi4 ships {cov.label} for {guide.element_text(cov.elements)} only.",
                    changes={"qm_method": f"{func}/{fix}"} if ok else {},
                    recommendation=f"Use {fix} (H-Rn without Ce-Lu)." if ok else "Leave them out of the QM stage.",
                    guide=(cov.key, "basis-sets"), evidence=guide.COVERAGE[cov.key].source))
    return out


def _spin_problems(ctx: _Ctx):
    open_shell = [(i, m) for i, m in ctx.mols if m.mult > 1]
    if not open_shell:
        return []
    out = []
    aim = [i for i, m in open_shell if ctx.engine[i] == "aimnet2"]
    if aim:
        out.append((aim, "elements.aimnet2-open-shell", "problem",
                    f"AIMNet2 would refine {len(aim)} open-shell molecule(s), but it is a closed-shell model.",
                    "Its model card says it 'assumes closed-shell electronic structure'; radicals need AIMNet2-NSE, "
                    "which QuantaBench does not use. Method auto does not check the spin."))
    ff = [i for i, m in open_shell if _family(ctx.engine[i]) in ("mmff", "gfnff")]
    if ff:
        out.append((ff, "elements.ff-open-shell", "warning",
                    f"{len(ff)} open-shell molecule(s) go to a force field, which does not model the spin.",
                    "MMFF94s and GFN-FF have no electrons: a radical is computed like a closed-shell molecule."))
    return out


def _common_method(ctx: _Ctx) -> str:
    """One installed refinement engine that takes every molecule of the batch in its phase ('' if none)."""
    order = (["aimnet2"] if not ctx.solvent else []) + list(SOLVENT_GFN2 if ctx.solvent else
                                                            ("gfn2-xtb", *SOLVENT_GFN2))
    for k in order:
        if k not in ctx.available:
            continue
        cov = guide.coverage_for_engine(k)
        if cov is not None and all(cov.handles(m.z, m.charge, m.mult)[0] for _i, m in ctx.mols):
            return k
    return ""


def _qm_findings(ctx: _Ctx) -> list[Finding]:
    cfg = ctx.cfg
    if not cfg.qm_method:
        return []
    func, basis = _qm_parts(cfg.qm_method)
    out = []
    every = [i for i, _m in ctx.mols]
    if not func:
        return [ctx.finding("problem", "QM level", "qm.level", f"QM level {cfg.qm_method!r} is not "
                            "'functional/basis'.", every, guide=("dft",))]
    if func.replace("-", "") in ("r2scan3c", "hf3c", "pbeh3c", "b973c", "wb97x3c"):
        out.append(ctx.finding(
            "warning", "QM level", "qm.3c", f"{func} is a composite with its own basis; the QM stage passes {basis}.",
            every, why="Psi4 runs the -3c methods with an empty basis (`energy('r2scan3c/')`); the QM stage takes "
                       "'functional/basis', so this combination is untested.", guide=("r2scan-3c",),
            recommendation="Use ωB97X-D/def2-SVPD in the QM stage; run r2SCAN-3c in Psi4 directly.",
            changes={"qm_method": "wb97x-d/def2-svpd"}, evidence="Psi4 1.9 manual, gCP interface"))
        return out
    anions = [i for i, m in ctx.mols if _anionic(m)]
    level = corrected_level(cfg, bool(anions))
    if not has_dispersion(func):
        fix = level.split("/", 1)[0]
        out.append(ctx.finding(
            "warning", "QM level", "qm.dispersion",
            f"{func} has no dispersion correction: folded conformers and stacked groups come out too high.", every,
            why="In the DFT conformer test D3(BJ) moved B3LYP's energy difference between Spartan's and GFN2's picks "
                "by +10 kJ/mol on average (−3 to +26) and took the correlation of its energies with the radius of "
                "gyration from −0.50 to 0.00; the Spartan B3LYP and PM6 results QuantaBench is checked against have "
                "no dispersion correction.",
            changes={"qm_method": level}, recommendation=f"Use {fix}" + (f" ({level})." if level != f"{fix}/{basis}"
                                                                         else "."),
            guide=("dispersion", guide.lookup(func).key if guide.lookup(func) else "dft"),
            evidence=EVIDENCE_DFT_TEST))
    if anions and not has_diffuse(basis):
        pf = sum(1 for i, m in ctx.mols if _anionic(m) and m.pfas)
        new_basis = level.split("/", 1)[1]
        out.append(ctx.finding(
            "warning", "QM level", "qm.diffuse",
            f"{len(anions)} molecule(s) with an anionic group ({pf} PFAS) and {basis}, a basis without diffuse "
            "functions: it squeezes the anion and favours compact conformers.", anions,
            why="Against r2SCAN-3c in PCM, ωB97X-D/def2-SVP put extended conformers too high by +15.0 ± 2.4 kJ/mol "
                "per Å of radius of gyration for PFAS ions and +7.8 ± 1.0 for non-fluorinated ions; def2-SVPD moved "
                "ωB97X-D to within 8 kJ/mol MAE of r2SCAN-3c, from 16.",
            changes={"qm_method": level},
            recommendation=f"Use {new_basis} (about 3x the time of {basis}).",
            guide=("def2-svpd", "basis-sets", "gcp"), evidence=f"{EVIDENCE_RANK1}; {EVIDENCE_RANK1B}"))
    elif basis in ("def2-svp", "6-31g*", "6-31g", "6-31g**", "def2-sv(p)"):
        out.append(ctx.finding(
            "info", "QM level", "qm.bsse",
            f"{basis} is a small basis: basis-set superposition error favours compact conformers a little.", every,
            why="For neutrals in the gas phase ωB97X-D/def2-SVP leaned +3.7 ± 1.9 kJ/mol per Å of radius of gyration "
                "toward compact conformers against def2-TZVP: 1-2 kJ/mol across their pools, no pick changed. "
                "r2SCAN-3c includes the gCP correction but is not selectable in the QM stage yet.",
            recommendation="Fine for screening; def2-TZVP (about 7x) for final energies.",
            guide=("gcp", "r2scan-3c", "def2-tzvp"), evidence=EVIDENCE_RANK1B))
    if ctx.solvent:
        if cfg.qm_task == "optimize":
            out.append(ctx.finding("info", "solvent", "qm.optimize-gas",
                                   "QM optimizations run in the gas phase; the PCM single point follows.", every,
                                   why="Psi4 has no analytic PCM gradients.", guide=("pcm", "single-point"),
                                   evidence="chem/qm.py"))
        alkyl_ions = [i for i, m in ctx.mols if m.ion and not m.pfas and m.alkyl >= 6]
        if alkyl_ions:
            out.append(ctx.finding(
                "warning", "solvent", "qm.pcm-electrostatic",
                f"{len(alkyl_ions)} surfactant-like ion(s) (an alkyl chain of 6 or more CH2/CH3) in IEF-PCM, which "
                "has no cavity or dispersion terms here.", alkyl_ions,
                why="RANK-1b: for alkyl carboxylates and alkylammonium the electrostatics-only reference favours "
                    "extended chains; a non-electrostatic term (SMD) would favour folded ones, so their DFT ranking "
                    "is uncertain.", guide=("pcm", "smd"), evidence=EVIDENCE_RANK1B,
                recommendation="Treat the conformer choice of these ions with care."))
    return out


def _solvent_findings(ctx: _Ctx) -> list[Finding]:
    from . import engines

    out = []
    if not ctx.solvent:
        ions = [i for i, m in ctx.mols if m.ion]
        if ions:
            multi = [i for i, m in ctx.mols if m.charge <= -2]
            zwit = sum(1 for i in ions if ctx.by_id[i].charge_class == "zwitterion")
            out.append(ctx.finding(
                "problem" if multi else "warning", "solvent", "solvent.gas-ions",
                f"{len(ions)} charged molecule(s) in the gas phase, where nothing screens their charge"
                + (f" ({len(multi)} multiply charged anion(s))." if multi else
                   f" ({zwit} zwitterion(s))." if zwit else "."), ions,
                why="Gas-phase ions fold around their charge, and a multiply charged anion can be unbound (its HOMO "
                    "above zero), its xTB polarizability then diverging"
                    + ("; a zwitterion such as an amino acid can transfer its proton back to the neutral form" if zwit
                       else "") + ". In water the refinement (ALPB), the xTB properties and the QM stage (PCM) "
                    "screen the charge; the charge state itself was chosen for water at the run's pH.",
                changes={"solvent": "water"}, recommendation="Run ions in water.",
                guide=("gas-phase", "solvation", "alpb"), evidence="method_guide: gas-phase"))
    else:
        gas_refine = [i for i, e in ctx.engine.items() if not engines.supports_solvent(e, ctx.solvent)]
        if gas_refine:
            names = Counter(ctx.engine[i] for i in gas_refine)
            fix = ctx.gfn2_engine(True)
            out.append(ctx.finding(
                "warning", "solvent", "solvent.gas-refinement",
                f"{len(gas_refine)} molecule(s) are refined in the gas phase in this {ctx.solvent} run: "
                + ", ".join(f"{k} ({n})" for k, n in names.most_common()) + " has no implicit solvent model.",
                gas_refine, why="The run's geometries, ranking and the engine's properties are then gas-phase ones; "
                                "tblite and the xtb program have ALPB.",
                changes={"method": fix} if fix else {},
                recommendation=f"Use {fix} for solvated geometries." if fix else "Install tblite for ALPB.",
                guide=("solvation", "alpb"), evidence="pipeline phase_steps"))
    return out


def _cost_findings(ctx: _Ctx, cost: CostEstimate) -> list[Finding]:
    cfg, out = ctx.cfg, []
    if cfg.conformer_search == "crest" and str(cfg.crest_method).lower() in ("gfn2", "gfn1"):
        big = [i for i, m in ctx.mols if m.heavy > 40 or m.rot > 15]
        if big:
            out.append(ctx.finding(
                "warning", "cost", "cost.crest-gfn2",
                f"CREST at {cfg.crest_method.upper()} on {len(big)} large or flexible molecule(s) (> 40 heavy atoms or "
                "> 15 rotatable bonds) takes hours each.", big,
                why="GFN2 searched 16 of 20 PFAS in 20,034 s on 6 lanes x 2 threads and the 4 largest hit the 3 h "
                    "limit; CREST with GFN-FF was more than 6x faster.",
                changes={"crest_method": "gfnff"}, recommendation="Search at GFN-FF; the refinement re-ranks.",
                guide=("crest-gfnff", "crest-gfn2"), evidence="docs/ENGINES.md"))
    big_ai = [i for i, m in ctx.mols if m.heavy > 100 and ctx.engine[i] == "aimnet2"]
    if big_ai:
        out.append(ctx.finding("warning", "cost", "cost.aimnet2-size",
                               f"{len(big_ai)} molecule(s) above 100 heavy atoms go to AIMNet2.", big_ai,
                               why="Its developers validated it up to about 100 heavy atoms (model card).",
                               guide=("aimnet2",), evidence="AIMNet2 model card"))
    if cfg.qm_method:
        slow = [(i, _qm_seconds(m, cfg.qm_method) * max(1, cfg.qm_top)) for i, m in ctx.mols]
        slow = [i for i, s in slow if s > 3600]
        if slow:
            out.append(ctx.finding(
                "warning", "cost", "cost.qm",
                f"The QM stage takes more than an hour each for {len(slow)} molecule(s) ({cfg.qm_top} single point(s) "
                f"at {cfg.qm_method}).", slow,
                why="One ωB97X-D/def2-SVP single point in PCM took 4-6 min for 30 heavy atoms and 12-27 min for 54 on a "
                    "4-thread NOVA lane, growing as nbf^2.9; a diffuse basis costs about 3x.",
                guide=("dft", "basis-sets"), evidence=EVIDENCE_DFT_TEST))
    if ctx.mols:
        out.append(ctx.finding("info", "cost", "cost.total",
                               f"Estimated time: {cost.text()}; the slowest molecule {cost.slowest_id} "
                               f"{_duration(cost.slowest_s)}.", [],
                               why="Measured timings on BTM NOVA scaled by molecule size; good to a factor of about 2. "
                                   + " ".join(cost.notes), evidence="docs/VALIDATION.md, ENGINES.md, PLATFORMS.md"))
    return out


def _search_findings(ctx: _Ctx) -> list[Finding]:
    cfg, out = ctx.cfg, []
    flexible = [i for i, m in ctx.mols if m.rot >= 8 and (m.ion or not m.hydrocarbon)]
    if flexible and cfg.refine_top < 20:
        out.append(ctx.finding(
            "warning", "conformer search", "search.refine-top",
            f"Only the lowest {cfg.refine_top} MMFF94s conformers are refined for {len(flexible)} flexible polar "
            "molecule(s) (8 or more rotatable bonds).", flexible,
            why="The pool is ranked by gas-phase MMFF94s first. Refining 5, QuantaBench's ensemble held the reference's "
                "lowest conformer in 35 of 120 neutral pools, and Spartan's pool had a GFN2-lower conformer for 49 of "
                "61 ions.", changes={"refine_top": 20}, recommendation="Refine the lowest 20 (the default).",
            guide=("rdkit-etkdg", "crest"), evidence=f"{EVIDENCE_RANK1B}; {EVIDENCE_VAL3}"))
    ions_flex = [i for i, m in ctx.mols if m.ion and m.rot >= 8]
    if ions_flex and ctx.solvent and cfg.conformer_search != "crest":
        crest_ok = "crest" in ctx.available
        out.append(ctx.finding(
            "info", "conformer search", "search.crest-ions",
            f"{len(ions_flex)} flexible ion(s) in water get their conformer pool from gas-phase MMFF94s.", ions_flex,
            why="The DFT re-ranking needs folded and extended conformers from RDKit and CREST; CREST searches in "
                "ALPB water.", changes={"conformer_search": "crest", "crest_method": "gfnff"} if crest_ok else {},
            recommendation=("CREST at GFN-FF (fast) widens the pool." if crest_ok else
                            "CREST (not found here) would widen the pool."),
            guide=("crest", "crest-gfnff"), evidence=EVIDENCE_DFT_TEST))
    macro = [i for i, m in ctx.mols if "macrocycle" in m.tags]
    if macro:
        out.append(ctx.finding(
            "info", "conformer search", "search.macrocycle",
            f"{len(macro)} macrocycle(s): large rings need more embedded conformers.", macro,
            why="ETKDGv3 has torsion terms for macrocycles (Wang et al. 2020), but a large ring has many low "
                "conformers; QuantaBench has not measured how many are enough.",
            changes={"max_conformers": max(cfg.max_conformers, 200)}, recommendation="Embed 200 or more, or use CREST.",
            guide=("rdkit-etkdg", "crest"), evidence="ETKDGv3 paper"))
    return out


def _reason(error: str) -> str:
    """'metal complex: keeping the largest fragment ... would remove Fe (...)' from an exception's text."""
    text = error.split(": ", 1)[-1] if error.split(": ", 1)[0].isidentifier() else error
    text = text.split(";", 1)[0].strip()
    return text if len(text) <= 160 else text[:157] + "..."


def _input_findings(ctx: _Ctx) -> list[Finding]:
    cfg, out = ctx.cfg, []
    if ctx.failed:
        reasons = Counter(_reason(m.error) for _i, m in ctx.failed)
        out.append(ctx.finding(
            "problem", "input", "input.not-computable",
            f"{len(ctx.failed)} record(s) cannot be built: " + "; ".join(r for r, _n in reasons.most_common(2)),
            [i for i, _m in ctx.failed], why="The pipeline's standardization or structure reading fails on them; "
                                              "they will be marked failed.",
            recommendation="Fix or remove these records.", evidence="chem/standardize.py"))
    multi = [i for i, m in ctx.mols if m.n_frags > 1]
    if multi:
        removed = Counter(f for i, m in ctx.mols for f in m.removed)
        if cfg.keep_largest_fragment:
            out.append(ctx.finding(
                "info", "input", "input.fragments",
                f"{len(multi)} record(s) have several fragments; the largest is computed and "
                + (", ".join(f"{f} ({n})" for f, n in removed.most_common(5)) or "the rest")
                + " removed (Keep largest fragment).", multi,
                why="A salt's counter-ion in the same gas-phase or continuum calculation would pair with the ion.",
                guide=("protonation",), evidence="chem/standardize.py"))
        else:
            out.append(ctx.finding(
                "warning", "input", "input.fragments-together",
                f"{len(multi)} record(s) with several fragments are computed together (Keep largest fragment is "
                "off): an ion pair or a mixture in one calculation.", multi,
                why="In the gas phase or a continuum a counter-ion sits on the ion; proton transfer or collapse can "
                    "follow.", changes={"keep_largest_fragment": True},
                recommendation="Turn Keep largest fragment on.", guide=("protonation",),
                evidence="chem/standardize.py"))
    many = [i for i, m in ctx.mols if m.undef_atoms >= 2 or m.undef_bonds]
    one = [i for i, m in ctx.mols if m.undef_atoms == 1 and not m.undef_bonds]
    if many:
        out.append(ctx.finding(
            "warning", "input", "input.stereo",
            f"{len(many)} molecule(s) with two or more undefined stereocentres or an undefined double bond: each "
            "embedded conformer gets a random configuration there, so one ensemble mixes diastereomers or E/Z "
            "isomers.", many,
            why="Diastereomers and E/Z isomers differ in energy and properties; RDKit embeds undefined centres at "
                "random.", recommendation="Define the stereochemistry in the input, or list each isomer.",
            guide=("rdkit-etkdg", "boltzmann"), evidence="RDKit ETKDG"))
    if one:
        out.append(ctx.finding(
            "info", "input", "input.stereo-one",
            f"{len(one)} molecule(s) with one undefined stereocentre: the ensemble mixes the two enantiomers.", one,
            why="Enantiomers have the same energy and properties in an achiral environment, so this is harmless.",
            guide=("rdkit-etkdg",), evidence="RDKit ETKDG"))
    rad = [i for i, m in ctx.mols if m.radicals]
    odd = [i for i, m in ctx.mols if m.mult > 1 and not m.radicals]
    if rad:
        out.append(ctx.finding(
            "warning", "input", "input.radicals",
            f"{len(rad)} record(s) are drawn as radicals and are computed as open shells.", rad,
            why="Radicals in database records are often drawing errors (a missing H or charge).",
            recommendation="Check the records.", guide=("protonation",), evidence="chem/standardize.py"))
    if odd:
        out.append(ctx.finding("warning", "input", "input.odd-electrons",
                               f"{len(odd)} molecule(s) have an odd electron count and are computed as doublets.",
                               odd, why="No radical is drawn; the charge may be wrong.", guide=("protonation",),
                               recommendation="Check their charge.", evidence="chem/standardize.py"))
    nplus = [i for i, m in ctx.mols if "sulfonamide_nplus" in m.tags]
    if nplus:
        out.append(ctx.finding(
            "problem", "input", "input.sulfonamide-nplus",
            f"{len(nplus)} molecule(s) carry a proton on a sulfonamide N.", nplus,
            why="One ion of the DFT conformer test was drawn so: the amine-protonated cation is 183 kJ/mol lower "
                "(ωB97X-D/PCM; GFN2 197, AIMNet2 173), and its 11.5 D dipole gap came from the proton, not the "
                "conformer.", changes={"protonate": True},
            recommendation="Turn protonation on (the rules move the proton to the most basic amine) or fix the record.",
            guide=("protonation",), evidence=f"{EVIDENCE_DFT_TEST}; {EVIDENCE_RANK1}"))
    drawn_nplus = [i for i, m in ctx.mols if "sulfonamide_nplus" in m.drawn_tags and "sulfonamide_nplus" not in m.tags]
    if drawn_nplus:
        out.append(ctx.finding(
            "info", "input", "input.sulfonamide-nplus-fixed",
            f"{len(drawn_nplus)} record(s) are drawn with a proton on a sulfonamide N; the protonation rules move it.",
            drawn_nplus, why="The amine-protonated tautomer was 183 kJ/mol lower in the DFT conformer test.",
            guide=("protonation",), evidence=EVIDENCE_DFT_TEST))
    if cfg.protonate and cfg.ph > 6.5:
        pfsa = [i for i, m in ctx.mols if "pf_sulfonamide" in m.tags and m.charge < 0]
        if pfsa:
            out.append(ctx.finding(
                "info", "input", "input.pf-sulfonamide",
                f"{len(pfsa)} perfluoroalkyl sulfonamide(s) are computed as anions at pH {cfg.ph:g} (rule pKa about "
                "6.5).", pfsa,
                why="Their N-H is acidic; near the pKa both forms are present and substituents shift it, so check "
                    "measured pKa values where they matter.", guide=("protonation",),
                evidence="chem/standardize.py, RULES"))
    if not cfg.protonate:
        ionizable = [i for i, m in ctx.mols if m.tags & {"carboxylic", "sulfonic", "phosphate", "amine"}
                     and not m.ion]
        if ionizable and ctx.solvent:
            out.append(ctx.finding(
                "info", "input", "input.no-protonation",
                f"Protonation is off: {len(ionizable)} acid(s) or base(s) are computed neutral in {ctx.solvent}.",
                ionizable, why="At pH 7 carboxylic and sulfonic acids are anions and aliphatic amines cations.",
                changes={"protonate": True}, recommendation="Turn protonation on for the species present in water.",
                guide=("protonation",), evidence="chem/standardize.py, RULES"))
    iso = [i for i, m in ctx.mols if "isotope" in m.tags]
    if iso:
        out.append(ctx.finding("info", "input", "input.isotopes", f"{len(iso)} molecule(s) carry isotope labels.",
                               iso, why="Isotopes change masses only: frequencies and thermochemistry, not energies.",
                               guide=("vibrations",) if "vibrations" in guide.ENTRIES else ()))
    return out


# ======================================================================================= coverage table
def coverage_rows(ctx: _Ctx) -> list[CoverageRow]:
    from . import engines

    sig = Counter((m.z, bool(m.charge), m.mult) for _i, m in ctx.mols)
    first_ids: dict[tuple, list[str]] = {}
    for i, m in ctx.mols:
        ids = first_ids.setdefault((m.z, bool(m.charge), m.mult), [])
        if len(ids) < ctx.max_ids:
            ids.append(i)
    inverse: dict[str, list[str]] = {}
    for k, gk in guide.ENGINE_ENTRIES.items():
        if k != "auto" and k in engines.all_engines():
            inverse.setdefault(gk, []).append(k)
    rows = []
    for key, cov in guide.COVERAGE.items():
        if key.startswith("crest-"):
            kind, installed = "conformer search", "crest" in ctx.available
        elif guide.ENTRIES[key].family == "Basis sets" or key == "r2scan-3c":
            kind, installed = "QM basis", "psi4" in ctx.available
        elif key in inverse:
            kind, installed = "refinement", any(k in ctx.available or k == "mmff" for k in inverse[key])
        else:
            kind, installed = "not in QuantaBench yet", None
        row = CoverageRow(key, cov.label, kind, installed=installed, note=cov.available)
        for (z, charged, mult), n in sig.items():
            if cov.missing(z):
                row.n_elements += n
                row.ids_excluded += first_ids[(z, charged, mult)][:max(0, ctx.max_ids - len(row.ids_excluded))]
            elif mult > 1 and cov.open_shell == "no":
                row.n_spin += n
                row.ids_excluded += first_ids[(z, charged, mult)][:max(0, ctx.max_ids - len(row.ids_excluded))]
            else:
                row.n_ok += n
                if mult > 1 and cov.open_shell == "ignored":
                    row.n_spin_ignored += n
        rows.append(row)
    return rows


# ======================================================================================= where each step runs
def phase_steps(ctx: _Ctx) -> list[tuple[str, str]]:
    """Where each stage computes in this run: ('conformer search', 'gas (MMFF94s ...)'), ..."""
    from . import engines, pipeline

    cfg = ctx.cfg
    if not ctx.solvent:
        steps = [("every step", "gas phase")]
        return steps
    engine = Counter(ctx.engine.values()).most_common(1)[0][0] if ctx.engine else cfg.method
    solvated = engines.supports_solvent(engine, ctx.solvent) if engine in engines.all_engines() else False
    label = pipeline.solvation_label(engine, ctx.solvent) if solvated else "gas (no implicit solvent model)"
    steps = [("conformer search", f"ALPB({ctx.solvent}) (CREST)" if cfg.conformer_search == "crest" else
              "gas (RDKit's MMFF94s has no solvent: the same pool in every phase)"),
             ("refinement", f"{label} ({engine})"),
             ("xTB properties", f"{pipeline.solvation_label('tblite-gfn1', ctx.solvent)}, tblite GFN1-xTB" if solvated
              and "tblite-gfn1" in ctx.available else "gas (no solvated refinement or no tblite)")]
    if cfg.frequencies:
        steps.append(("frequencies", label if solvated and engines.plugin(engine) is not None else "gas"))
    if cfg.qm_method:
        steps.append(("QM stage", f"PCM({ctx.solvent}) single points (IEF-PCM, electrostatics only)"
                      + ("; optimizations in the gas phase" if cfg.qm_task == "optimize" else "")))
    return steps


# ======================================================================================= presets
PRESET_NAMES = {"fast": "Fast screen", "recommended": "Recommended", "careful": "Careful"}


def _presets(ctx: _Ctx) -> list[Preset]:
    """Fast screen, Recommended and Careful for this batch in its phase. Every method in them takes every molecule
    (elements, charge, spin): AIMNet2 only when no molecule is open-shell, the QM stage only when the def2 basis sets
    cover every element."""
    cfg = ctx.cfg
    mols = [m for _i, m in ctx.mols]
    n_ion = sum(m.ion for m in mols)
    n_anion = sum(m.charge < 0 or bool(m.neg) for m in mols)
    n_polar = sum(not m.ion and not m.hydrocarbon for m in mols)
    n_open = sum(m.mult > 1 for m in mols)
    polar_or_ion = n_ion + n_polar
    water = bool(ctx.solvent)
    phase = f"in {ctx.solvent}" if water else "in the gas phase"
    all_aimnet = bool(mols) and all(_aimnet_ok(m) for m in mols)
    if water:
        method = ctx.gfn2_engine(True) or "auto"
    elif "aimnet2" in ctx.available and not n_open:
        method = "aimnet2" if all_aimnet else "auto"
    else:
        method = ctx.gfn2_engine(False) or "auto"
    aimnet = method == "aimnet2" or (method == "auto" and "aimnet2" in ctx.available and not water)
    m_name = {"auto": "auto (AIMNet2 where its elements allow, else GFN2-xTB)" if aimnet else
              "auto (the best installed engine per molecule)", "aimnet2": "AIMNet2"}.get(
        method, f"GFN2-xTB ({method})" if method in GFN2 else method)
    def2_ok = all(not guide.COVERAGE["def2-svp"].missing(m.z) for m in mols)
    basis = "def2-svpd" if n_anion else "def2-svp"
    crest_here = "crest" in ctx.available
    needs_qm = [] if "psi4" in ctx.available else ["Psi4"]

    # ---- Fast screen
    fast = {"method": method, "conformer_search": "rdkit", "max_conformers": 30, "refine_top": 10,
            "energy_window_kcal": 6.0, "qm_method": "", "frequencies": False}
    fast_why = (f"RDKit conformers, the lowest 10 refined with {m_name} {phase}, no DFT: geometries and trend "
                "descriptors for the whole batch, quickly.")
    if water and polar_or_ion:
        fast_why += (f" The conformer each of the {polar_or_ion} polar or charged molecules ends up with is rough: "
                     "GFN2-xTB/ALPB ranks polar molecules in water at 0.6-0.7 of a random pick's regret and PFAS ions "
                     "no better than a random pick (RANK-1b), so treat their conformer-dependent descriptors (dipole, "
                     "shape) as approximate.")
    elif aimnet:
        fast_why += " AIMNet2 ranks neutral gas-phase conformers as well as anything cheap measured (RANK-1b)."

    # ---- Recommended
    rec = {"method": method, "conformer_search": "rdkit", "max_conformers": 50, "refine_top": 20,
           "energy_window_kcal": 6.0, "qm_method": "", "qm_top": 1}
    rec_needs: list[str] = []
    if water and polar_or_ion and def2_ok:
        rec.update(energy_window_kcal=15.0, qm_method=f"wb97x-d/{basis}", qm_top=8, qm_task="energy")
        rec_needs = needs_qm
        rec_why = (f"{m_name} gives solvated geometries and xTB properties. Then the QM stage re-ranks the lowest 8 "
                   f"conformers with ωB97X-D/{_basis_name(basis)} in PCM water, from a pool kept up to 15 kcal/mol "
                   "above GFN2's lowest: the DFT re-ranking the validation recommends for polar and charged molecules "
                   "in water, whose GFN2-xTB/ALPB ranking is unreliable. ωB97X-D includes the dispersion folded "
                   "conformers need"
                   + ("; def2-SVPD's diffuse functions hold the anions' loosely bound electrons, which def2-SVP "
                      "squeezes toward compact conformers (RANK-1b)." if n_anion else ".")
                   + " Until the rank step (BACKLOG RANK-2: g-xTB + CPCM at about 0.1 % of the DFT cost) lands, only "
                   "the qm_* descriptors follow the DFT-lowest conformer.")
    elif water and polar_or_ion:
        rec_why = (f"{m_name} {phase}, the lowest 20 refined. The DFT re-ranking the validation recommends for polar "
                   "and charged molecules in water is left out: the def2 basis sets do not cover every element here.")
    elif water:
        rec_why = (f"{m_name} {phase}, the lowest 20 refined: for pure hydrocarbons GFN2-xTB ranks conformers as well "
                   "as any cheap method measured (12 of 12, RANK-1b), so no DFT re-ranking is needed.")
    else:
        rec_why = f"{m_name} {phase}, the lowest 20 of 50 RDKit conformers refined."
        if mols and all(m.hydrocarbon for m in mols):
            rec_why += (" For pure hydrocarbons every cheap method ranks conformers within 4 kJ/mol of r2SCAN-3c "
                        "(GFN2-xTB 12 of 12, RANK-1b), so no DFT re-ranking is needed.")
        elif aimnet:
            rec_why += (" AIMNet2 is the best cheap gas-phase ranker measured for neutrals (39 of 40 non-fluorinated "
                        "and 18 of 20 PFAS within 4 kJ/mol of r2SCAN-3c), and neutral dipoles hardly depend on the "
                        "conformer, so no DFT re-ranking is needed.")
        else:
            rec_why += (" AIMNet2, the best measured gas-phase ranker for neutrals, is not used: "
                        + ("some molecules are open-shell." if n_open else "it is not installed here."))
        if n_ion:
            rec_why += f" The {n_ion} ion(s) belong in water (see the solvent finding): gas-phase ions are unscreened."

    # ---- Careful
    careful = dict(rec, refine_top=30)
    careful_needs = list(rec_needs)
    if crest_here:
        careful.update(conformer_search="crest", crest_method="gfnff", crest_quick=True, max_conformers=50)
        search = "CREST's metadynamics search at GFN-FF" + (" in ALPB water" if water else "")
    else:
        careful.update(conformer_search="rdkit", max_conformers=200)
        search = "200 RDKit conformers (CREST, which would search wider, is not found here)"
    if water and polar_or_ion and def2_ok:
        careful.update(qm_method="wb97x-d/def2-svpd", qm_top=12, energy_window_kcal=15.0)
        careful_needs = needs_qm
        careful_why = (f"{search} widens the pool with folded and extended conformers; the lowest 30 are refined with "
                       f"{m_name}, then ωB97X-D/def2-SVPD in PCM re-ranks the lowest 12 (dispersion, and diffuse "
                       "functions for anions and polarizabilities). The most defensible ranking QuantaBench runs "
                       "today, and the most expensive.")
    elif not water and def2_ok:
        level = "def2-svpd" if n_anion else "def2-tzvp"
        careful.update(qm_method=f"wb97x-d/{level}", qm_top=3)
        careful_needs = needs_qm
        careful_why = (f"{search} widens the pool, {m_name} refines the lowest 30, and ωB97X-D/{_basis_name(level)} "
                       "single points on the lowest 3 give DFT-quality properties ("
                       + ("diffuse functions for the anions)." if n_anion else
                          "a triple-zeta basis removes most of def2-SVP's superposition error).")
                       + " The ranking stays the refinement's: for neutrals in the gas phase the measured cheap "
                       "rankers are close to DFT.")
    else:
        careful_why = (f"{search} widens the pool and {m_name} refines the lowest 30; no DFT stage (the def2 basis sets "
                       "do not cover every element here).")

    out = []
    for key, changes, why, needs in (("fast", fast, fast_why, []), ("recommended", rec, rec_why, rec_needs),
                                     ("careful", careful, careful_why, careful_needs)):
        pcfg = replace(cfg, **changes)
        cost = estimate_cost(ctx.mols, pcfg, _Resolver(pcfg, ctx.solvent, ctx.available))
        diff = {k: v for k, v in changes.items() if getattr(cfg, k) != v}
        out.append(Preset(key, PRESET_NAMES[key], changes, why, cost, needs=sorted(set(needs)), config=pcfg,
                          diff=diff))
    return out


def review(inputs, cfg, *, available: Iterable[str] | None = None, max_ids: int = MAX_IDS,
           progress: Callable[[int, int], None] | None = None,
           should_stop: Callable[[], bool] | None = None) -> Review:
    """The review of ``inputs`` (``MoleculeInput``s) for a run with ``cfg`` (``BuildConfig``). ``available``: the
    engine keys installed here plus "psi4" and "crest" (default: ``probe_available()``; tests pass their own)."""
    t0 = time.perf_counter()
    inputs = list(inputs)
    mols = molecule_features(inputs, cfg, progress=progress, should_stop=should_stop)
    prof = build_profile(mols, cfg, max_ids)
    avail = set(probe_available() if available is None else available)
    ctx = _Ctx(mols, cfg, avail, max_ids)
    cost = estimate_cost(ctx.mols, cfg, ctx.resolve)
    findings = (_input_findings(ctx) + _element_findings(ctx) + _rank_findings(ctx) + _search_findings(ctx)
                + _solvent_findings(ctx) + _qm_findings(ctx) + _cost_findings(ctx, cost))
    findings.sort(key=lambda f: (LEVEL_ORDER[f.level], -f.n))
    presets = _presets(ctx)
    from . import qm as _qm

    settings = {"phase": ctx.phase, "solvent": ctx.solvent, "method": cfg.method, "ph": cfg.ph,
                "protonate": cfg.protonate, "keep_largest_fragment": cfg.keep_largest_fragment,
                "conformer_search": cfg.conformer_search, "crest_method": cfg.crest_method,
                "max_conformers": cfg.max_conformers, "refine_top": cfg.refine_top,
                "energy_window_kcal": cfg.energy_window_kcal,
                "qm_method": _qm.level_string(cfg.qm_method) if cfg.qm_method and _qm_parts(cfg.qm_method)[0] else
                cfg.qm_method, "qm_top": cfg.qm_top}
    return Review(n=len(inputs), settings=settings, profile=prof, coverage=coverage_rows(ctx), findings=findings,
                  presets=presets, recommended="recommended", cost=cost, steps=phase_steps(ctx),
                  methods=dict(Counter(ctx.engine.values()).most_common()), seconds=time.perf_counter() - t0)


# ======================================================================================= text
#: BuildConfig field -> the build option that sets it (None: Settings only)
CLI_FLAGS = {"method": "--method", "solvent": "--solvent", "conformer_search": "--conformers",
             "crest_method": "--crest-method", "max_conformers": "--max-confs", "refine_top": "--refine-top",
             "energy_window_kcal": "--window", "qm_method": "--qm", "qm_top": "--qm-top", "qm_task": "--qm-task",
             "ph": "--ph", "frequencies": "--freq"}


def cli_options(changes: dict) -> str:
    """The build options for ``changes`` ('--method aimnet2 --qm wb97x-d/def2-svpd ...')."""
    parts = []
    for k, v in changes.items():
        if k == "crest_quick":
            if not v:
                parts.append("--crest-full")
            continue
        if k == "protonate":
            parts.append("--protonate" if v else "--no-protonate")
            continue
        if k == "frequencies":
            parts.append("--freq" if v else "--no-freq")
            continue
        flag = CLI_FLAGS.get(k)
        if flag is None:
            parts.append(f"(Settings: {k} = {v})")
        elif k == "qm_method" and not v:
            continue
        elif k == "solvent":
            parts.append(f"{flag} {v or 'none'}")
        else:
            parts.append(f"{flag} {v:g}" if isinstance(v, float) else f"{flag} {v}")
    return " ".join(parts)


def _wrap(text: str, width: int, indent: str, first: str = "") -> list[str]:
    import textwrap

    return textwrap.wrap(text, width, initial_indent=first or indent, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False) or [first.rstrip()]


def _ids(ids: list[str], n: int, max_ids: int) -> str:
    shown = ", ".join(ids[:max_ids])
    return shown + (f", ... ({n - min(n, max_ids)} more)" if n > max_ids else "")


def settings_line(r: Review) -> str:
    s = r.settings
    search = "CREST " + str(s["crest_method"]).upper() if s["conformer_search"] == "crest" else "RDKit ETKDGv3 + MMFF94s"
    methods = ", ".join(f"{k} {n}" for k, n in r.methods.items())
    return (f"{s['phase'] if s['phase'] == 'gas' else s['solvent']} phase, pH {s['ph']:g}"
            + ("" if s["protonate"] else " (protonation off)") + f", method {s['method']}"
            + (f" -> {methods}" if methods else "") + f", {search}, "
            + (f"QM {s['qm_method']} on {s['qm_top']}" if s["qm_method"] else "no QM stage"))


def render_text(r: Review, width: int = 100, max_ids: int = 8) -> str:
    p = r.profile
    lines = [f"Review of {r.n:,} molecule{'s' if r.n != 1 else ''} ({settings_line(r)})", ""]

    def row(label: str, groups: list[Group], limit: int = 30) -> None:
        if not groups:
            return
        text = ", ".join(f"{g.label} {g.n:,}" for g in groups[:limit])
        lines.extend(_wrap(text, width, " " * 18, f"  {label:<16}"))

    lines.append("Batch")
    row("Elements", p.sections.get("elements", []))
    row(f"Charge at pH {r.settings['ph']:g}" if r.settings.get("protonate") else "Charge as drawn",
        p.sections.get("charge", []))
    row("Net charge", p.sections.get("net_charge", []))
    pf = p.group("pfas", "pfas")
    chains = ", ".join(f"{g.label} {g.n:,}" for g in p.sections.get("chain", []))
    lines.extend(_wrap(f"{pf.n:,} of {p.n:,}" + (f"; perfluoroalkyl chain {chains}" if chains else "")
                       + f" ({PFAS_RULE})", width, " " * 18, f"  {'PFAS':<16}"))
    row("Groups", p.sections.get("groups", []))
    st = p.stats
    if st["heavy"]["median"] is not None:
        lines.extend(_wrap(
            f"heavy atoms median {st['heavy']['median']:g} ({st['heavy']['min']}-{st['heavy']['max']}): "
            + ", ".join(f"{g.label} {g.n:,}" for g in p.sections.get("heavy", []))
            + f"; rotatable bonds median {st['rotatable']['median']:g} ({st['rotatable']['min']}-"
              f"{st['rotatable']['max']}): " + ", ".join(f"{g.label} {g.n:,}" for g in p.sections.get("rotatable", [])),
            width, " " * 18, f"  {'Size':<16}"))
    flags = {g.key: g for g in p.sections.get("flags", [])}
    shown = ["hydrocarbon", "open_shell", "macrocycle", "undefined_stereo", "undefined_double", "multi_fragment",
             "isotope", "metal", "unusual", "not_computable"]
    lines.extend(_wrap(", ".join(f"{FLAG_LABELS[k]} {flags[k].n if k in flags else 0:,}" for k in shown), width,
                       " " * 18, f"  {'Other':<16}"))
    row("Removed", p.sections.get("removed", []), limit=8)
    lines.append("")

    lines.append("Methods that can take these molecules (elements, charge, spin)")
    lines.append(f"  {'method':<30}{'kind':<24}{'takes':>10}  here")
    for c in r.coverage:
        here = "-" if c.installed is None else "installed" if c.installed else "not found"
        takes = f"{c.n_ok:,}/{p.n - p.count('flags', 'not_computable'):,}"
        extra = []
        if c.n_elements:
            extra.append(f"{c.n_elements:,} lack elements")
        if c.n_spin:
            extra.append(f"{c.n_spin:,} open shell")
        if c.n_spin_ignored:
            extra.append(f"spin ignored for {c.n_spin_ignored:,}")
        lines.append(f"  {c.label:<30}{c.kind:<24}{takes:>10}  {here}" + (f"  ({'; '.join(extra)})" if extra else ""))
    lines.append("")

    lines.append("Where each step runs")
    for step, where in r.steps:
        lines.append(f"  {step:<18}{where}")
    lines.append("")

    counts = r.counts()
    lines.append(f"Findings ({counts['problem']} problems, {counts['warning']} warnings, {counts['info']} notes)")
    for f in r.findings:
        tag = {"problem": "PROBLEM", "warning": "WARNING", "info": "note"}[f.level]
        lines.extend(_wrap(f"{f.topic}: {f.message}", width, " " * 11, f"  {tag:<9}"))
        if f.ids:
            lines.extend(_wrap(f"Molecules: {_ids(f.ids, f.n, max_ids)}", width, " " * 13, " " * 11))
        if f.why:
            lines.extend(_wrap(f"Why: {f.why}" + (f" ({f.evidence})" if f.evidence else ""), width, " " * 13,
                               " " * 11))
        if f.recommendation or f.changes:
            rec = f.recommendation + (f" [{cli_options(f.changes)}]" if f.changes else "")
            lines.extend(_wrap(f"Do: {rec.strip()}", width, " " * 13, " " * 11))
        if f.guide:
            lines.extend(_wrap("Read: quantabench-cli methods " + ", ".join(f.guide), width, " " * 13, " " * 11))
    lines.append("")

    lines.append("Presets")
    for pr in r.presets:
        lines.extend(_wrap(f"{pr.name}: {pr.cost.text()}", width, " " * 4, "  "))
        lines.extend(_wrap(pr.rationale, width, " " * 4))
        opts = cli_options(pr.diff)
        lines.extend(_wrap(f"Options: {opts or '(the settings reviewed)'}", width, " " * 6, " " * 4))
        if pr.needs:
            lines.append(f"    Needs: {', '.join(pr.needs)} (not found on this machine)")
    lines.append("")
    rec = r.preset(r.recommended)
    if rec is not None:
        lines.extend(_wrap(f"Recommended configuration: quantabench-cli build ... {cli_options(rec.diff)}".rstrip(),
                           width, "  ", ""))
    lines.append(f"(review took {r.seconds:.1f} s; quantabench-cli methods NAME explains any method)")
    return "\n".join(lines)


def render_header(r: Review, max_findings: int = 6, width: int = 100) -> str:
    """A few lines before a build: what the batch is and the problems and warnings."""
    p = r.profile
    cls = ", ".join(f"{g.n:,} {g.key}" for g in p.sections.get("charge", []))
    counts = r.counts()
    lines = [f"Review: {r.n:,} molecules ({cls}; {p.group('pfas', 'pfas').n:,} PFAS) - {counts['problem']} problems, "
             f"{counts['warning']} warnings"]
    shown = [f for f in r.findings if f.level != "info"][:max_findings]
    for f in shown:
        tag = "PROBLEM" if f.level == "problem" else "WARNING"
        lines.extend(_wrap(f"{f.message}" + (f" [{cli_options(f.changes)}]" if f.changes else ""), width, " " * 11,
                           f"  {tag:<9}"))
    rest = sum(1 for f in r.findings if f.level != "info") - len(shown)
    if rest > 0:
        lines.append(f"  ... and {rest} more")
    lines.append("  quantabench-cli review (same inputs and options) explains each one and suggests presets; "
                 "--no-review skips this.")
    return "\n".join(lines)
