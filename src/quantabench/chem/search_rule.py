"""Which conformer search a molecule gets (CONF-AUTO, 2026-11-16).

``BuildConfig.conformer_search``:

    "rdkit"   RDKit ETKDGv3 + MMFF94s for every molecule (the default)
    "crest"   CREST iMTD-GC for every molecule
    "auto"    per molecule: CREST for a neutral molecule with at least ``crest_min_rotors`` rotatable bonds (default
              9), RDKit for the rest

Why "auto" (the conformer checks of Nov 13-15, ``Runs/conformer-checks-2026-11-13/report_full.md``):

- Up to 8 rotatable bonds the RDKit search converged: the tranche's conformer was the best known for 55 of 68.
- From 9 on it did not, and CREST (GFN-FF, quick) found a lower conformer for most neutral molecules: the best known
  for 10 of 23 in gas (RDKit's 3) and the lower of the two for 16 of 21 in ALPB water (RDKit's 6).
- For ions it was worse: two anions ended 108 and 281 kJ/mol above the best known in gas (GFN-FF and charged chains).
  So ions and zwitterions keep RDKit's search.

``crest_with_rdkit``: where CREST runs (asked for, or chosen by "auto"), the RDKit search runs too and the refinement
starts from both pools (``merge_pools``): in water RDKit still held a conformer more than 5 kJ/mol lower for 5 of 21.

The charge class and the rotatable-bond count are the ones the model advisor (``chem.advisor``) and the conformer
checks used: the net charge of the prepared molecule (at the run's pH), "zwitterion" when it carries both a positive
and a negative atom that are not a bonded pair (a nitro group's N+ and O- are a pair); RDKit's strict rotatable bonds
on the heavy-atom graph.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

SEARCHES = ("rdkit", "crest", "auto")
DEFAULT_MIN_ROTORS = 9
#: the result's ``conformer_search`` when both pools were refined
MERGED = "crest+rdkit"
_STRICT = rdMolDescriptors.NumRotatableBondsOptions.Strict
_CHARGED = Chem.MolFromSmarts("[!+0]")


def check(search: str) -> str:
    if search not in SEARCHES and search != "":
        raise ValueError(f"conformer_search must be one of {', '.join(SEARCHES)}, not {search!r}")
    return search or "rdkit"


def charge_class(mol: Chem.Mol, charge: int | None = None) -> str:
    """"neutral", "anion", "cation" or "zwitterion" (net 0 with charged atoms that are not a +/- bonded pair).
    ``charge``: the molecule's total charge when the input sets it; default the formal charge of ``mol``."""
    q = Chem.GetFormalCharge(mol) if charge is None else int(charge)
    if q < 0:
        return "anion"
    if q > 0:
        return "cation"
    pos = neg = False
    for (i,) in mol.GetSubstructMatches(_CHARGED):
        a = mol.GetAtomWithIdx(i)
        f = a.GetFormalCharge()
        if any(nb.GetFormalCharge() * f < 0 for nb in a.GetNeighbors()):    # a nitro-type pair
            continue
        pos, neg = pos or f > 0, neg or f < 0
    return "zwitterion" if pos and neg else "neutral"


def rotatable_bonds(mol: Chem.Mol) -> int:
    """RDKit's strict rotatable bonds of the heavy-atom graph (explicit hydrogens would count methyl rotors)."""
    from rdkit import rdBase

    with rdBase.BlockLogs():
        try:
            heavy = Chem.RemoveHs(mol)
        except Exception:  # noqa: BLE001 - a structure RDKit cannot sanitize: its graph as it is
            heavy = Chem.RemoveHs(mol, sanitize=False)
            heavy.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(heavy)
        return int(rdMolDescriptors.CalcNumRotatableBonds(heavy, _STRICT))


def flexible_neutral(charge_cls: str, rotors: int, min_rotors: int = DEFAULT_MIN_ROTORS) -> bool:
    return charge_cls == "neutral" and int(rotors) >= int(min_rotors)


def decide(cfg, mol: Chem.Mol | None, charge: int | None = None) -> tuple[bool, str]:
    """(CREST for this molecule?, why: "" unless the search is "auto"). ``mol``: the prepared molecule."""
    search = check(getattr(cfg, "conformer_search", "rdkit"))
    if search != "auto":
        return search == "crest", ""
    if mol is None:
        return False, "conformer search auto: RDKit (no structure to judge)"
    cls, rot = charge_class(mol, charge), rotatable_bonds(mol)
    need = int(getattr(cfg, "crest_min_rotors", DEFAULT_MIN_ROTORS))
    if flexible_neutral(cls, rot, need):
        return True, f"conformer search auto: CREST (neutral, {rot} rotatable bonds >= {need})"
    why = cls if cls != "neutral" else f"{rot} rotatable bond{'s' if rot != 1 else ''} < {need}"
    return False, f"conformer search auto: RDKit ({why})"


def crest_for(cfg, mol: Chem.Mol | None, charge: int | None = None) -> bool:
    return decide(cfg, mol, charge)[0]


def may_use_crest(cfg) -> bool:
    """Whether a run of ``cfg`` may run CREST at all (planning lanes, the cost model, the engines a run records)."""
    return getattr(cfg, "conformer_search", "rdkit") in ("crest", "auto")


def used_crest(search: str) -> bool:
    """A result's ``conformer_search`` ("crest", "crest+rdkit", "rdkit", "imported") came from CREST."""
    return str(search or "").startswith("crest")


def merged(cfg, method: str = "") -> bool:
    """Whether a CREST molecule of ``cfg`` also gets RDKit's pool. Not with the refinement "mmff": it takes the pool's
    own energies, and CREST's (GFN-FF) and MMFF94s's are not on one scale."""
    return bool(getattr(cfg, "crest_with_rdkit", False)) and method != "mmff"


def describe(cfg) -> str:
    """One line for logs and summaries."""
    search = check(getattr(cfg, "conformer_search", "rdkit"))
    both = " (+ RDKit's pool)" if getattr(cfg, "crest_with_rdkit", False) else ""
    if search == "rdkit":
        return "RDKit ETKDGv3 + MMFF94s"
    if search == "crest":
        return f"CREST{both}"
    need = int(getattr(cfg, "crest_min_rotors", DEFAULT_MIN_ROTORS))
    return f"auto: CREST{both} for neutral molecules with >= {need} rotatable bonds, RDKit for the rest"


def merge_pools(crest_set, rdkit_set, refine_top: int, rmsd_threshold: float = 0.5):
    """One ``ConformerSet`` with the lowest ``refine_top`` of CREST's pool (by CREST's energies) and the lowest
    ``refine_top`` of RDKit's (by MMFF94s's), RDKit's conformers that duplicate a CREST one dropped. The two energy
    scales are never compared: the refinement takes every conformer of the merged set (``refine_count``) and ranks
    them at its own level. Duplicates are found by heavy-atom RMSD after alignment on the same atom order (fast); a
    symmetry-related duplicate is refined twice and removed by the re-pruning after the refinement."""
    from rdkit.Chem import rdMolAlign

    from .conformers import ConformerSet

    n = max(1, int(refine_top))
    out = Chem.Mol(crest_set.mol)
    out.RemoveAllConformers()
    ids, energies = [], []
    for cid, e in zip(crest_set.conf_ids[:n], crest_set.energies_kcal[:n]):
        ids.append(out.AddConformer(Chem.Conformer(crest_set.mol.GetConformer(cid)), assignId=True))
        energies.append(e)
    n_crest = len(ids)
    heavy = [a.GetIdx() for a in out.GetAtoms() if a.GetAtomicNum() > 1]
    amap = list(zip(heavy, heavy))
    dropped = 0
    rd_mol = rdkit_set.mol
    for cid, e in zip(rdkit_set.conf_ids[:n], rdkit_set.energies_kcal[:n]):
        new = out.AddConformer(Chem.Conformer(rd_mol.GetConformer(cid)), assignId=True)
        dup = False
        for ref in ids[:n_crest]:
            if rdMolAlign.AlignMol(out, out, prbCid=new, refCid=ref, atomMap=amap) < rmsd_threshold:
                dup = True
                break
        if dup:
            out.RemoveConformer(new)
            dropped += 1
            continue
        ids.append(new)
        energies.append(e)
    notes = list(crest_set.notes) + list(rdkit_set.notes)
    notes.append(f"conformer pools merged: CREST's lowest {n_crest} and RDKit's lowest {len(ids) - n_crest}"
                 + (f" ({dropped} duplicate{'s' if dropped != 1 else ''} of a CREST conformer dropped)" if dropped
                    else "") + "; all are refined")
    return ConformerSet(mol=out, conf_ids=ids, energies_kcal=energies,
                        forcefield=f"{crest_set.forcefield} + {rdkit_set.forcefield}",
                        n_embedded=crest_set.n_embedded + rdkit_set.n_embedded, notes=notes, refine_count=len(ids))
