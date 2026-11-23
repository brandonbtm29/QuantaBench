"""Salts as separate entries (SETTINGS-1 follow-up; ``BuildConfig.counterion_entries``, a science setting).

A build computes one species per record: the largest fragment, after the build filters (``standardize``). With
*Keep counter-ions as separate entries* on, the counter-ions it removes become molecules of their own in the same run,
so a QSAR table of potassium PFOS has the PFOS anion and K+ side by side:

    id               ``<parent id>_<formula>`` ("kpfos_K+", "tbatfa_C16H36N+"), unique in the run
    name             the counter-ion's name (``standardize.NAMED_COUNTERIONS``: "potassium"), else its formula
    counterion_of    the ids of every record of the run it came from, in input order ("kpfos;kpfhxs"): a
                     summary.csv column (``COUNTERION_OF`` in the input's meta)
    posted fields    copied from the first of those records (File > Open's posted SD fields / table columns)

Which removed fragments count: an ion (a formal charge as found after the cleanup) or a named counter-ion (its neutral
form too: HCl is chloride); never water, a copy of the kept fragment (salt stoichiometry) or a neutral molecule.
Entries are de-duplicated by the standard InChIKey of the fragment as drawn, within the run and against the run's own
inputs: a hundred potassium salts give one K+. The expansion depends only on the input list and the settings, so every
machine of a grid build computes the same entries (``jobs.run_resumable`` expands before choosing its chunk).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from . import import_options as imp
from . import standardize
from .inputs import MoleculeInput

#: ``MoleculeInput.meta`` key of a counter-ion entry: the ids of the records it came from (a list)
COUNTERION_OF = "qb_counterion_of"
#: longest id a worker accepts (``dist.worker`` labels: 60 characters)
MAX_ID = 60
_NAMES = standardize.FragmentStrip(counterions=tuple(standardize.NAMED_COUNTERIONS))


def _charged(frag: Chem.Mol) -> bool:
    return any(a.GetFormalCharge() for a in frag.GetAtoms())


def counterions_of(mol: Chem.Mol, keep_largest: bool, strip: standardize.FragmentStrip | None = None
                   ) -> list[tuple[str, str]]:
    """(SMILES, name) of the counter-ions the build removes from ``mol`` (a record as drawn), in the order it removes
    them; [] for one fragment or a structure the build cannot compute (``UnsupportedStructure``)."""
    if len(Chem.GetMolFrags(mol)) < 2:
        return []
    info: dict = {}
    try:
        with rdBase.BlockLogs():
            parent = standardize.standardize(Chem.Mol(mol), keep_largest_fragment=keep_largest, neutralize=False,
                                             info=info, strip=strip)
    except (standardize.UnsupportedStructure, ValueError, RuntimeError):
        return []
    with rdBase.BlockLogs():
        parent_key = Chem.MolToSmiles(Chem.RemoveHs(parent, sanitize=False))
    out = []
    for smiles in info.get("removed_fragments") or []:
        with rdBase.BlockLogs():
            frag = Chem.MolFromSmiles(smiles)
        if frag is None or smiles == parent_key or smiles == "O":
            continue
        atoms = tuple(range(frag.GetNumAtoms()))
        name = _NAMES.why(frag, atoms)
        if name or _charged(frag):
            out.append((smiles, name))
    return out


def _formula(frag: Chem.Mol) -> str:
    """'K+', 'Cl-', 'Ca+2', 'C16H36N+' (RDKit's formula with the charge): the id suffix."""
    return rdMolDescriptors.CalcMolFormula(frag)


def expand(inputs: Sequence[MoleculeInput], cfg, log: Callable[[str], None] | None = None) -> list[MoleculeInput]:
    """``inputs`` followed by one entry per distinct counter-ion the build removes from them (module docstring), when
    ``cfg.counterion_entries`` is on; ``inputs`` unchanged otherwise. Unknown counter-ion names in the build filters
    raise ``ValueError`` (as the build does)."""
    inputs = list(inputs)
    if not getattr(cfg, "counterion_entries", False):
        return inputs
    strip = standardize.FragmentStrip.of(cfg)
    keep = bool(getattr(cfg, "keep_largest_fragment", True))
    found: dict[str, dict] = {}           # InChIKey -> {"smiles", "name", "parents", "first"}
    for inp in inputs:
        if (inp.meta or {}).get(COUNTERION_OF):
            continue                      # an entry made earlier (a resume with the same list)
        try:
            mol = inp.to_mol()
        except ValueError:
            continue
        for smiles, name in counterions_of(mol, keep, strip):
            with rdBase.BlockLogs():
                frag = Chem.MolFromSmiles(smiles)
            key, _why = standardize.inchikey(frag)
            key = key or f"smiles:{smiles}"
            entry = found.setdefault(key, {"smiles": smiles, "name": name, "parents": [], "first": inp})
            if inp.id not in entry["parents"]:
                entry["parents"].append(inp.id)
    if not found:
        return inputs
    present: set[str] = set()             # the run's own single-fragment inputs: an ion given as a molecule
    for inp in inputs:
        try:
            mol = inp.to_mol()
        except ValueError:
            continue
        if len(Chem.GetMolFrags(mol)) == 1 and mol.GetNumHeavyAtoms() <= 40:
            key, _why = standardize.inchikey(mol)
            if key:
                present.add(key)
    taken = {x.id.casefold() for x in inputs}
    out = list(inputs)
    added = 0
    for key, e in found.items():
        if key in present:
            continue
        first: MoleculeInput = e["first"]
        with rdBase.BlockLogs():
            suffix = _formula(Chem.MolFromSmiles(e["smiles"]))
        base = f"{first.id[:MAX_ID - len(suffix) - 1]}_{suffix}"
        ident, n = base, 2
        while ident.casefold() in taken:
            ident = f"{base[:MAX_ID - len(str(n)) - 1]}_{n}"
            n += 1
        taken.add(ident.casefold())
        meta: dict = {COUNTERION_OF: list(e["parents"])}
        posted = (first.meta or {}).get(imp.FIELDS)
        if isinstance(posted, dict) and posted:
            meta[imp.FIELDS] = dict(posted)
        out.append(MoleculeInput(id=ident, smiles=e["smiles"], name=e["name"] or suffix, source=first.source,
                                 meta=meta))
        added += 1
    if added and log is not None:
        log(f"Counter-ions as separate entries: {added} added ({', '.join(x.id for x in out[len(inputs):][:5])}"
            f"{', ...' if added > 5 else ''})")
    return out
