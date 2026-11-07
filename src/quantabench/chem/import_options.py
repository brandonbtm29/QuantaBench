"""Import options (UI-4): what happens to structures between the file and the molecule list, modeled on Spartan's SDF
Import Options dialog. ``inputs.read_file(..., options=ImportOptions(...))`` applies them; the GUI reads a file
first (``inputs.read_raw``), shows the choices the file has (``survey``), then calls ``apply``.

Fixup
    ``convert_2d``: a 2D structure (MOL/SDF with the 2D flag or every z = 0, ChemDraw, 2D Marvin/CML) gets 3D
    coordinates (ETKDGv3 + MMFF94s). ``grow_hydrogens``: explicit hydrogens with coordinates. ``keep_coords``: the
    file's 3D coordinates become the first conformer of the build (``pipeline`` seeds the conformer search with them)
    instead of only new ETKDG conformers.
Additional data
    ``post_fields``: SD properties and table columns become columns of the molecule table (and summary.csv);
    ``parse_fields``: numbers are parsed, a unit after the number goes into the column name ("E (kJ/mol)").
    ``id_field`` / ``name_field``: which field gives the id and the name ("" = automatic).
Filters (each removal is recorded per molecule: ``meta[NOTES]`` and ``meta[REMOVED]``, which the build reports)
    ``remove_water``; ``remove_acids`` (HF, HCl, HBr, HI and other small neutral inorganic fragments);
    ``strip_counterions`` (alkali and alkaline-earth cations, small inorganic ions such as Cl-, SO4 2-, NH4+);
    ``keep_largest`` (only the fragment the build would keep: ``standardize`` picks it the same way). Fragments with
    other metals are never removed here: the build decides about them (it refuses metal complexes drawn as ions).
    The build's own *Keep largest fragment* (Settings > Build defaults) still applies to what the import kept.
Labeling
    ``label_order``: "read" (atoms in the file's order) or "standard" (heavy atoms by RDKit canonical rank, then each
    hydrogen after the atoms before it), so atom numbers do not depend on the program that wrote the file.
Several structures of one constitution
    ``same_constitution``: "molecules" (each is a molecule; repeats of an id get _2, _3) or "conformers" (one
    molecule with one conformer per structure, as Spartan documents hold them).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, replace

from rdkit import Chem, rdBase

from .formats import COORDS, READER

#: ``MoleculeInput.meta`` keys written here (``formats`` writes READER and COORDS)
NOTES = "qb_import_notes"         # what the import did to the molecule (copied into the build's notes)
REMOVED = "qb_import_removed"     # SMILES of the fragments the filters removed (the build adds them to removed_fragments)
FILE_CHARGE = "qb_file_charge"    # formal charge as drawn in the file, before the filters
FIELDS = "qb_fields"              # posted fields: {column name: value} (numbers parsed with parse_fields)
KEEP_COORDS = "qb_keep_coords"    # True: the build uses the 3D coordinates as its first conformer(s)
CONFORMER_SOURCES = "qb_conformer_sources"   # records merged into one molecule as conformers
#: meta keys that are not data fields of the file
INTERNAL_KEYS = {"smiles_column", "row", "pubchem_cid"}

LABEL_ORDERS = ("read", "standard")
SAME_CONSTITUTION = ("molecules", "conformers")
#: the filter names of ``--filter`` (CLI) and their option fields; "hcl" is Spartan's name for the acid filter
FILTERS = {"water": "remove_water", "acids": "remove_acids", "hcl": "remove_acids",
           "counterions": "strip_counterions", "largest": "keep_largest"}


@dataclass
class ImportOptions:
    convert_2d: bool = True
    grow_hydrogens: bool = True
    keep_coords: bool = False
    post_fields: bool = False
    parse_fields: bool = True
    id_field: str = ""
    name_field: str = ""
    remove_water: bool = True
    remove_acids: bool = True
    strip_counterions: bool = False
    keep_largest: bool = False
    label_order: str = "read"
    same_constitution: str = "molecules"

    def __post_init__(self) -> None:
        if self.label_order not in LABEL_ORDERS:
            raise ValueError(f"label_order must be one of {', '.join(LABEL_ORDERS)}, not {self.label_order!r}")
        if self.same_constitution not in SAME_CONSTITUTION:
            raise ValueError(f"same_constitution must be one of {', '.join(SAME_CONSTITUTION)}, "
                             f"not {self.same_constitution!r}")

    # ---------------------------------------------------------------- settings
    @classmethod
    def from_settings(cls, s) -> ImportOptions:
        """From ``Settings.import_`` (``settings.ImportSettings``), or a whole ``Settings``. Bad stored values give
        the defaults."""
        s = getattr(s, "import_", s)
        values = {f.name: getattr(s, f.name) for f in fields(cls) if hasattr(s, f.name)}
        try:
            return cls(**values)
        except (TypeError, ValueError):
            good = {k: v for k, v in values.items() if k not in ("label_order", "same_constitution")}
            return cls(**good)

    def to_settings(self, s) -> None:
        """Store these choices in ``Settings.import_`` (or the ``ImportSettings`` given)."""
        s = getattr(s, "import_", s)
        for f in fields(self):
            if hasattr(s, f.name):
                setattr(s, f.name, getattr(self, f.name))

    # ---------------------------------------------------------------- command line
    def with_text(self, text: str) -> ImportOptions:
        """These options changed by ``key=value,key=value`` or a JSON object (``--import-options``). Keys are the
        field names (``keep_coords``, ``label_order``, ...); ``filter=water+acids`` sets the four filters at once."""
        text = (text or "").strip()
        if not text:
            return self
        if text.startswith("{"):
            try:
                pairs = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"--import-options: not valid JSON ({exc})") from None
            if not isinstance(pairs, dict):
                raise ValueError("--import-options: a JSON object is expected")
        else:
            pairs = {}
            for part in re.split(r"[,;]", text):
                if not part.strip():
                    continue
                key, sep, value = part.partition("=")
                if not sep:
                    raise ValueError(f"--import-options: {part.strip()!r} is not key=value")
                pairs[key.strip()] = value.strip()
        out = self
        for key, value in pairs.items():
            key = key.strip().lower().replace("-", "_")
            if key in ("filter", "filters"):
                out = out.with_filters(value if isinstance(value, str) else ",".join(value))
                continue
            names = {f.name: f for f in fields(self)}
            if key not in names:
                raise ValueError(f"--import-options: unknown option {key!r} (options: {', '.join(names)}, filter)")
            default = getattr(ImportOptions(), key)
            if isinstance(default, bool):
                value = _bool(value, key)
            else:
                value = str(value)
            out = replace(out, **{key: value})
        return out

    def with_filters(self, spec: str) -> ImportOptions:
        """``water,acids`` (also ``hcl``, ``counterions``, ``largest``; ``none`` for no filter): exactly these
        filters on, the others off."""
        names = [p.strip().lower() for p in re.split(r"[,+ ]", spec or "") if p.strip()]
        unknown = [n for n in names if n not in FILTERS and n != "none"]
        if unknown:
            raise ValueError(f"unknown filter(s) {', '.join(unknown)} (filters: water, acids or hcl, counterions, "
                             "largest, none)")
        on = {FILTERS[n] for n in names if n != "none"}
        return replace(self, **{f: f in on for f in set(FILTERS.values())})

    def filters(self) -> list[str]:
        return [name for name, f in FILTERS.items() if name != "hcl" and getattr(self, f)]


def _bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "y"):
        return True
    if text in ("0", "false", "no", "off", "n"):
        return False
    raise ValueError(f"--import-options: {key} must be true or false, not {value!r}")


# ================================================================================================ fragments
def _fragment_smiles(mol: Chem.Mol, atoms: Iterable[int]) -> str:
    with rdBase.BlockLogs():
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), isomericSmiles=True)


def fragment_kind(mol: Chem.Mol, atoms: tuple[int, ...]) -> str:
    """"water", "acid" (HF, HCl, HBr, HI, small neutral inorganic fragments), "counterion" (alkali and
    alkaline-earth cations, small inorganic ions) or "" for the fragment of ``mol`` made of ``atoms``."""
    from .standardize import COUNTERION_METALS, METALS

    heavy = [mol.GetAtomWithIdx(i) for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    if not heavy:
        return ""
    charge = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)
    nums = [a.GetAtomicNum() for a in heavy]
    if len(heavy) == 1 and nums[0] in COUNTERION_METALS:
        return "counterion" if charge > 0 else ""
    if 6 in nums or any(z in METALS for z in nums) or len(heavy) > 5:
        return ""
    if charge != 0:
        return "counterion"
    hydrogens = sum(a.GetTotalNumHs() for a in heavy) + sum(
        1 for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 1)
    if nums == [8] and hydrogens == 2:
        return "water"
    return "acid"


def _has_kept_metal(mol: Chem.Mol, atoms: tuple[int, ...]) -> bool:
    """A metal the build must decide about (anything but a lone alkali or alkaline-earth ion)."""
    from .standardize import COUNTERION_METALS, METALS

    heavy = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    return any(z in METALS for z in heavy) and not (len(heavy) == 1 and heavy[0] in COUNTERION_METALS)


def _largest_fragment(mol: Chem.Mol, frags: list[tuple[int, ...]]) -> int:
    """Index of the fragment ``standardize`` keeps (RDKit's FragmentParent: most atoms, then weight, then SMILES)."""
    from rdkit.Chem.MolStandardize import rdMolStandardize

    with rdBase.BlockLogs():
        parent = rdMolStandardize.FragmentParent(Chem.RemoveHs(mol, sanitize=False), skipStandardize=True)
        key = Chem.MolToSmiles(parent)
    labels = [Chem.MolToSmiles(Chem.MolFromSmiles(_fragment_smiles(mol, f)) or Chem.Mol()) for f in frags]
    if key in labels:
        return labels.index(key)
    return max(range(len(frags)), key=lambda i: sum(1 for j in frags[i] if mol.GetAtomWithIdx(j).GetAtomicNum() > 1))


def filter_fragments(mol: Chem.Mol, options: ImportOptions) -> tuple[Chem.Mol, list[str], list[str]]:
    """(``mol`` without the fragments the filters remove, removed fragment SMILES, notes). Atom order, coordinates
    and properties of what is kept stay as they are. The last fragment is never removed."""
    frags = list(Chem.GetMolFrags(mol))
    if len(frags) < 2:
        return mol, [], []
    wanted = {"water": options.remove_water, "acid": options.remove_acids, "counterion": options.strip_counterions}
    drop: dict[int, str] = {}
    for i, f in enumerate(frags):
        kind = fragment_kind(mol, f)
        if kind and wanted[kind]:
            drop[i] = kind
    notes: list[str] = []
    if options.keep_largest:
        keep = _largest_fragment(mol, frags)
        metal = [i for i in range(len(frags)) if i != keep and _has_kept_metal(mol, frags[i])]
        if metal:
            notes.append("keep largest fragment not applied on import: "
                         f"{', '.join(_fragment_smiles(mol, frags[i]) for i in metal)} holds a metal (the build "
                         "decides: it does not compute a metal complex drawn as separate ions)")
        else:
            drop.update({i: drop.get(i, "smaller fragment") for i in range(len(frags)) if i != keep})
    if len(drop) >= len(frags):            # never remove everything: keep the largest of the dropped
        keep = _largest_fragment(mol, frags)
        drop.pop(keep, None)
    if not drop:
        return mol, [], notes
    removed = [_fragment_smiles(mol, frags[i]) for i in sorted(drop)]
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for i in drop:
        for a in frags[i]:
            rw.RemoveAtom(a)
    rw.CommitBatchEdit()
    out = rw.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(out)
    names = {"water": "water", "acid": "acid / small inorganic", "counterion": "counter-ion",
             "smaller fragment": "smaller fragment"}
    counts = Counter((removed[k], names[drop[i]]) for k, i in enumerate(sorted(drop)))
    notes.append("removed on import: " + ", ".join(f"{n} x {s} ({why})" if n > 1 else f"{s} ({why})"
                                                    for (s, why), n in counts.items()))
    return out, removed, notes


# ================================================================================================ coordinates
def embed_2d(mol: Chem.Mol, grow_hydrogens: bool, seed: int = 0xF00D) -> tuple[Chem.Mol | None, str]:
    """3D coordinates for a 2D structure: ETKDGv3, then MMFF94s when it has parameters. Stereo drawn with wedges is
    kept (RDKit perceived it when reading). Without ``grow_hydrogens`` the hydrogens added for the embedding are
    removed again (the file's own explicit H stay). (None, why) when RDKit cannot embed it."""
    from rdkit.Chem import AllChem

    n = mol.GetNumAtoms()
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    with rdBase.BlockLogs():
        try:
            ok = AllChem.EmbedMolecule(mh, params) == 0
            if not ok:
                params.useRandomCoords = True
                ok = AllChem.EmbedMolecule(mh, params) == 0
        except RuntimeError as exc:
            return None, f"RDKit could not embed 3D coordinates ({str(exc).splitlines()[0] if str(exc) else exc})"
        if not ok:
            return None, "RDKit could not embed 3D coordinates"
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, mmffVariant="MMFF94s", maxIters=500)
    if not grow_hydrogens and mh.GetNumAtoms() > n:
        rw = Chem.RWMol(mh)
        rw.BeginBatchEdit()
        for i in range(n, mh.GetNumAtoms()):
            rw.RemoveAtom(i)
        rw.CommitBatchEdit()
        mh = rw.GetMol()
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mh)
    mh.GetConformer().Set3D(True)
    return mh, ""


def grow_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Explicit hydrogens, placed from the heavy atoms' coordinates when there are any."""
    with rdBase.BlockLogs():
        return Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)


def standard_order(mol: Chem.Mol) -> Chem.Mol:
    """Atoms renumbered in a standard order: heavy atoms by RDKit canonical rank, then the hydrogens, each group of
    H in the order of the atoms they are bonded to. Coordinates and properties follow the atoms."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    heavy = sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1), key=lambda i: ranks[i])
    pos = {i: k for k, i in enumerate(heavy)}

    def h_key(i: int):
        nbs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()]
        return (pos.get(nbs[0], len(heavy)) if nbs else len(heavy), ranks[i])

    hydrogens = sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1), key=h_key)
    order = heavy + hydrogens
    if order == list(range(mol.GetNumAtoms())):
        return mol
    out = Chem.RenumberAtoms(mol, order)
    for k, v in mol.GetPropsAsDict(includePrivate=True, includeComputed=False).items():
        if not out.HasProp(k):
            out.SetProp(k, str(v))
    return out


def _conformer_onto(base: Chem.Mol, other: Chem.Mol) -> Chem.Conformer | None:
    """``other``'s coordinates on ``base``'s atoms (same constitution, maybe another atom order), or None."""
    if other.GetNumConformers() == 0 or base.GetNumAtoms() != other.GetNumAtoms():
        return None
    match = other.GetSubstructMatch(base, useChirality=False)
    if len(match) != base.GetNumAtoms():
        return None
    src = other.GetConformer()
    conf = Chem.Conformer(base.GetNumAtoms())
    for i, j in enumerate(match):
        conf.SetAtomPosition(i, src.GetAtomPosition(j))
    conf.Set3D(src.Is3D())
    return conf


# ================================================================================================ fields
_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµμ°%Å][A-Za-z0-9µμ°%Å/·*^\-²³ ]{0,15})?\s*$")


def parse_value(text: str) -> tuple[object, str]:
    """(number or the text, unit): "12.5" -> (12.5, ""), "-3 kJ/mol" -> (-3, "kJ/mol"); CAS numbers, dates,
    numbers with leading zeros (ids) and anything else stay text."""
    text = str(text).strip()
    m = _NUMBER.match(text)
    if not m:
        return text, ""
    num = m.group(1)
    digits = num.lstrip("+-")
    if len(digits) > 1 and digits[0] == "0" and digits[1].isdigit():
        return text, ""                     # "007", "0123": an id, not a number
    value: object = float(num)
    if re.fullmatch(r"[-+]?\d+", num):
        value = int(num)
    return value, (m.group(2) or "").strip()


def field_names(meta: dict) -> list[str]:
    """The data fields of a record's ``meta`` (SD properties, table columns), in file order."""
    from .inputs import ID_NOTE, SMILES_COLUMNS

    return [k for k in meta if not str(k).startswith("qb_") and k != ID_NOTE and k not in INTERNAL_KEYS
            and str(k).strip().lower() not in SMILES_COLUMNS]


def posted_fields(meta: dict, parse: bool = True) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in field_names(meta):
        value = meta[k]
        if parse:
            value, unit = parse_value(value)
            if unit and "(" not in k and "[" not in k:
                k = f"{k} ({unit})"
        out[str(k)] = value
    return out


def _lookup(meta: dict, name: str) -> str:
    lower = {str(k).strip().lower(): k for k in meta}
    key = lower.get(name.strip().lower())
    return str(meta[key]).strip() if key is not None else ""


# ================================================================================================ survey
@dataclass
class Survey:
    """What a file offers the Import Options dialog: the counts decide which choices are shown."""

    n: int = 0
    n_2d: int = 0
    n_3d: int = 0
    fields: list[str] = field(default_factory=list)
    fragments: Counter = field(default_factory=Counter)   # "water" | "acid" | "counterion" | "other" (mixture) -> molecules
    n_multi_fragment: int = 0
    n_same_constitution: int = 0      # molecules that share their constitution with another one
    readers: list[str] = field(default_factory=list)

    def has_choices(self) -> bool:
        return bool(self.n_2d or self.fields or self.n_multi_fragment or self.n_same_constitution)


def survey(items) -> Survey:
    from .inputs import _parses

    s = Survey(n=len(items))
    keys: Counter = Counter()
    for it in items:
        kind = it.meta.get(COORDS, "")
        s.n_2d += kind == "2d"
        s.n_3d += kind == "3d"
        for k in field_names(it.meta):
            if k not in s.fields:
                s.fields.append(k)
        reader = it.meta.get(READER)
        if reader and reader not in s.readers:
            s.readers.append(reader)
        if it.mol is not None and it.mol.GetNumConformers():
            keys[it.smiles] += 1
        smi = it.smiles or ""
        if "." not in smi:
            continue
        mol = it.mol if it.mol is not None else (Chem.MolFromSmiles(smi) if _parses(smi) else None)
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            s.n_multi_fragment += 1
            kinds = [fragment_kind(mol, f) or "other" for f in frags]
            for k in dict.fromkeys(kinds):      # "other": a mixture (two or more organic fragments)
                if k != "other" or kinds.count("other") > 1:
                    s.fragments[k] += 1
    s.n_same_constitution = sum(n for n in keys.values() if n > 1)
    return s


# ================================================================================================ apply
def apply(items, options: ImportOptions, log: Callable[[str], None] | None = None) -> list:
    """``items`` (``MoleculeInput`` as read, ``inputs.read_raw``) with ``options`` applied; ids made unique. Each
    molecule's ``meta[NOTES]`` says what was done to it; one summary line per kind of change goes to ``log``."""
    from .inputs import ID_NOTE, dedupe_ids, safe_id
    from .standardize import formal_charge

    out = []
    counts: Counter = Counter()
    for it in items:
        it = replace(it, meta=dict(it.meta))
        notes = list(it.meta.get(NOTES) or [])
        mol = it.mol
        from_smiles = mol is None and bool(it.smiles)
        if from_smiles and "." in it.smiles:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(it.smiles)
        if mol is not None:
            charge = formal_charge(mol)
            mol, removed, why = filter_fragments(mol, options)
            notes += why
            if removed:
                it.meta[REMOVED] = list(it.meta.get(REMOVED) or []) + removed
                it.meta.setdefault(FILE_CHARGE, charge)
                counts["filtered"] += 1
        if from_smiles:
            if mol is not None and it.meta.get(REMOVED):
                it.smiles = Chem.MolToSmiles(mol)
            mol = None
        if mol is not None:
            kind = it.meta.get(COORDS, "")
            if kind == "2d" and options.convert_2d:
                embedded, why = embed_2d(mol, options.grow_hydrogens)
                if embedded is None:
                    notes.append(f"2D structure not converted to 3D: {why}")
                else:
                    mol = embedded
                    it.meta[COORDS] = "3d"
                    notes.append("2D structure converted to 3D (ETKDGv3 + MMFF94s) on import")
                    counts["2d"] += 1
            elif options.grow_hydrogens and any(a.GetTotalNumHs() for a in mol.GetAtoms()):   # H not yet atoms
                mol = grow_hydrogens(mol)
                counts["hydrogens"] += 1
            if options.label_order == "standard":
                mol = standard_order(mol)
                notes.append("atoms numbered in standard order (RDKit canonical ranks)")
            if options.keep_coords and it.meta.get(COORDS) == "3d":
                it.meta[KEEP_COORDS] = True
            elif KEEP_COORDS in it.meta and not options.keep_coords:
                it.meta.pop(KEEP_COORDS)
            it.mol = mol
            with rdBase.BlockLogs():
                it.smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if options.id_field:
            value = _lookup(it.meta, options.id_field)
            if value:
                it.id = safe_id(value, fallback=it.id)
            elif ID_NOTE not in it.meta:          # read_sdf has already said so
                notes.append(f"no {options.id_field} field: id {it.id!r} kept")
        if options.name_field:
            value = _lookup(it.meta, options.name_field)
            if value:
                it.name = value
        if options.post_fields:
            posted = posted_fields(it.meta, options.parse_fields)
            if posted:
                it.meta[FIELDS] = posted
        if notes:
            it.meta[NOTES] = notes
        out.append(it)
    if options.same_constitution == "conformers":
        out = _merge_conformers(out, counts)
    else:
        out = _number_repeats(out)
    warnings: list[str] = []
    out = dedupe_ids(out, warnings)
    lines = {"filtered": "{} molecule(s): fragments removed by the import filters ({})",
             "2d": "{} 2D structure(s) converted to 3D", "hydrogens": "{} structure(s): hydrogens added",
             "merged": "{} structure(s) of a constitution already read became conformers of that molecule"}
    for key, text in lines.items():
        if counts[key]:
            (log or _noop)(text.format(counts[key], ", ".join(options.filters()) or "none"))
    for w in warnings:
        (log or _noop)(w)
    return out


def _noop(_msg: str) -> None:
    pass


def _number_repeats(items: list) -> list:
    """Structures that repeat an id and a constitution with other coordinates are molecules of their own: they
    get _2, _3 (``dedupe_ids`` would drop them as exact repeats). Exact repeats (same coordinates) are left for
    ``dedupe_ids`` to drop."""
    seen: dict[tuple[str, str], list[tuple]] = {}
    out = []
    for it in items:
        if it.mol is not None and it.mol.GetNumConformers():
            key = (it.id.casefold(), it.smiles)
            geometry = tuple(round(float(x), 3) for x in it.mol.GetConformer().GetPositions().ravel())
            shapes = seen.setdefault(key, [])
            if geometry not in shapes:
                shapes.append(geometry)
                if len(shapes) > 1:
                    it = replace(it, id=f"{it.id}_{len(shapes)}")
        out.append(it)
    return out


def _merge_conformers(items: list, counts: Counter) -> list:
    """One molecule per constitution (canonical SMILES), with every structure's coordinates as a conformer."""
    first: dict[str, object] = {}
    out = []
    for it in items:
        if it.mol is None or it.mol.GetNumConformers() == 0:
            out.append(it)
            continue
        base = first.get(it.smiles)
        if base is None:
            first[it.smiles] = it
            out.append(it)
            continue
        conf = _conformer_onto(base.mol, it.mol)
        if conf is None:
            out.append(it)
            continue
        if base.mol.GetConformer().Is3D() and conf.Is3D():
            base.mol.AddConformer(conf, assignId=True)
        base.meta[CONFORMER_SOURCES] = list(base.meta.get(CONFORMER_SOURCES) or [base.source]) + [it.source]
        counts["merged"] += 1
    for it in first.values():
        sources = it.meta.get(CONFORMER_SOURCES)
        if sources:
            it.meta[NOTES] = list(it.meta.get(NOTES) or []) + [
                f"{len(sources)} structures of this constitution in the file read as {it.mol.GetNumConformers()} "
                f"conformer(s) of one molecule ({', '.join(sources[:5])}{', ...' if len(sources) > 5 else ''})"]
    return out
