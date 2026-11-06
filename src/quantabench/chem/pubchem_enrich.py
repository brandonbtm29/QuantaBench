"""Enrich a molecule list with PubChem: CIDs, computed properties, CAS numbers, structure validation and
experimental values (``quantabench-cli pubchem enrich``). UI-agnostic; the network client is ``chem.pubchem``.

CONTRACT:

    read_queries(path, id_col=None, smiles_col=None, sheet=None) -> [Query]   # CSV/Excel, SDF, run folder, summary.csv
    enrich(queries, client, experimental=False, validate=False, progress=None) -> Enrichment
    compare_structures(qb_keys, pubchem_inchikey) -> (outcome, note)
    write_csv(path, rows, columns=None)

Resolution, per molecule, in this order: an explicit CID column; a CAS RN (the ``--id-col`` value, a CAS column,
or an id/name that is a valid CAS RN) through PubChem's name index, cross-checked with the CIDs' registry numbers
(xrefs RN); the InChIKey as drawn (``input_inchikey``, or ``inchikey`` in a plain table); the SMILES; the name or
DTXSID. A CAS RN that gives several CIDs keeps those that list it as a registry number; if several remain it is
``ambiguous`` (``pubchem_cids`` lists them; the one whose InChIKey equals QuantaBench's structure is used when
there is one). Requests are batched where PUG REST allows it (properties per 100 CIDs, synonyms and registry
numbers per 50); identifier lookups and PUG View records are one request each.

Structure validation (``--validate``) compares QuantaBench's structure for a molecule with PubChem's structure
for the molecule's CAS RN, by standard InChIKey (``chem.validate``): ``match`` (identical), ``stereo_differs``
(same connectivity and protonation, other stereo or isotopes), ``charge_or_protonation_differs`` (same first
block, or the same after removing counter-ions and neutralizing: a salt and its parent acid),
``component_of_mixture`` (PubChem's record for the CAS RN has several components and QuantaBench's structure is
one of them: a mixture or a salt with an organic counter-ion; the note names the others and says when QuantaBench
kept a smaller component than the largest), ``different_structure``, ``cas_not_in_pubchem``, ``ambiguous_cas``
(several CIDs carry the CAS RN). Rows without
a CAS RN or without a structure get ``no_cas`` or ``no_structure``. QuantaBench's structure is the input as drawn
(``input_inchikey``, or the SMILES / SD record), then the computed species (``inchikey``: desalted, protonated);
the best agreement counts and ``pubchem_match_note`` says which key matched.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import pubchem as pc

CAS_COLUMNS = ("cas", "casrn", "cas_rn", "cas no.", "cas no", "cas_no", "cas number", "cas_number", "cas-rn",
               "cas registry number", "cas_candidates")
ID_COLUMNS = ("id", "dtxsid", "label_resolved")      # then the CAS RN, then the name
NAME_COLUMNS = ("name", "preferred_name", "chemical_name", "chemical name", "compound", "title")
SMILES_COLUMNS = ("input_smiles", "smiles", "qsar_ready_smiles", "ms_ready_smiles", "canonical_smiles",
                  "isomeric_smiles")
CID_COLUMNS = ("cid", "pubchem_cid")
DTXSID_COLUMNS = ("dtxsid",)
OUTCOMES = ("match", "stereo_differs", "charge_or_protonation_differs", "component_of_mixture", "different_structure",
            "cas_not_in_pubchem", "ambiguous_cas", "no_cas", "no_structure", "error")

#: pubchem_* columns of the enriched table (after the input's own columns)
PUBCHEM_COLUMNS = (
    "pubchem_status", "pubchem_query", "pubchem_query_type", "pubchem_cid", "pubchem_cids", "pubchem_title",
    "pubchem_cas", "pubchem_cas_all", "pubchem_cas_in_xrefs", "pubchem_formula", "pubchem_mw", "pubchem_exact_mass",
    "pubchem_smiles", "pubchem_connectivity_smiles", "pubchem_inchi", "pubchem_inchikey", "pubchem_xlogp",
    "pubchem_tpsa", "pubchem_complexity", "pubchem_hbd", "pubchem_hba", "pubchem_rotatable_bonds", "pubchem_charge",
    "pubchem_heavy_atoms", "pubchem_note")
VALIDATE_COLUMNS = ("qb_inchikey", "pubchem_match", "pubchem_match_note")
EXPERIMENTAL_COLUMNS = (
    "pubchem_exp_records", "pubchem_st_records", "pubchem_st_pure_mN_m", "pubchem_st_pure_T_C",
    "pubchem_st_pure_source", "pubchem_st_aqueous_min_mN_m", "pubchem_st_unclassified_mN_m", "pubchem_st_review",
    "pubchem_density_g_cm3", "pubchem_density_T_C", "pubchem_density_class", "pubchem_mp_C", "pubchem_bp_C",
    "pubchem_parachor")
_PROP_MAP = (("pubchem_title", "Title"), ("pubchem_formula", "MolecularFormula"), ("pubchem_mw", "MolecularWeight"),
             ("pubchem_exact_mass", "ExactMass"), ("pubchem_smiles", "SMILES"),
             ("pubchem_connectivity_smiles", "ConnectivitySMILES"), ("pubchem_inchi", "InChI"),
             ("pubchem_inchikey", "InChIKey"), ("pubchem_xlogp", "XLogP"), ("pubchem_tpsa", "TPSA"),
             ("pubchem_complexity", "Complexity"), ("pubchem_hbd", "HBondDonorCount"),
             ("pubchem_hba", "HBondAcceptorCount"), ("pubchem_rotatable_bonds", "RotatableBondCount"),
             ("pubchem_charge", "Charge"), ("pubchem_heavy_atoms", "HeavyAtomCount"))


# ============================================================================================ inputs
@dataclass
class Query:
    id: str
    cas: str = ""
    name: str = ""
    smiles: str = ""
    inchikey: str = ""              # the structure as drawn
    species_inchikey: str = ""      # QuantaBench's computed species (desalted, protonated)
    cid: int | None = None
    mol: object | None = None       # RDKit molecule (SD records)
    row: dict = field(default_factory=dict)


def _lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).lower().strip(): c for c in columns if c}


def _col(columns: Iterable[str], wanted: Sequence[str], explicit: str | None = None) -> str | None:
    low = _lower_map(columns)
    if explicit:
        key = explicit.lower().strip()
        if key not in low:
            raise ValueError(f"Column {explicit!r} not in the input (columns: {', '.join(map(str, columns))})")
        return low[key]
    return next((low[w] for w in wanted if w in low), None)


def _cell(v) -> str:
    from .inputs import _cell as cell

    return cell(v)


def _read_rows(path: Path, sheet=None) -> list[dict]:
    """Rows of a CSV/Excel table, or of a QuantaBench run folder / merged tables (summary.csv, paired.csv)."""
    if path.is_dir():
        for name in ("summary.csv", "paired.csv"):
            if (path / name).is_file():
                return _read_rows(path / name)
        raise FileNotFoundError(f"{path}: no summary.csv or paired.csv (a QuantaBench run folder or merged tables)")
    ext = path.suffix.lower()
    if ext == ".csv":
        csv.field_size_limit(1 << 30)
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    if ext in (".xlsx", ".xlsm", ".xls"):
        from .inputs import _excel_rows

        _cols, rows = _excel_rows(path, sheet)
        return [{k: _cell(v) for k, v in r.items()} for _n, _i, r in rows]
    raise ValueError(f"Not a table, SD file or run folder: {path.name}")


def queries_from_rows(rows: list[dict], id_col: str | None = None, smiles_col: str | None = None,
                      source: str = "") -> list[Query]:
    if not rows:
        return []
    cols = list(rows[0])
    icol = _col(cols, ID_COLUMNS, id_col)
    ccol = _col(cols, CAS_COLUMNS)
    ncol = _col(cols, NAME_COLUMNS)
    scol = _col(cols, SMILES_COLUMNS, smiles_col)
    cidcol = _col(cols, CID_COLUMNS)
    dcol = _col(cols, DTXSID_COLUMNS)
    low = _lower_map(cols)
    qb_table = "input_inchikey" in low       # QuantaBench summary/merged/paired table
    ik_drawn = low.get("input_inchikey") or (None if qb_table else low.get("inchikey"))
    ik_species = low.get("inchikey") if qb_table else None
    out = []
    for n, row in enumerate(rows, 1):
        cells = {k: _cell(v) for k, v in row.items() if k}
        if not any(cells.values()):
            continue
        ident = cells.get(icol, "") if icol else ""
        cas = pc.normalize_cas(cells.get(icol, "")) if id_col and icol else ""
        if not cas and ccol:
            cas = pc.normalize_cas(cells.get(ccol, ""))
        if not cas:
            cas = next((pc.normalize_cas(cells.get(c, "")) for c in (icol, ncol) if c
                        and pc.is_cas(cells.get(c, ""))), "")
        cid = None
        if cidcol and cells.get(cidcol, "").isdigit():
            cid = int(cells[cidcol])
        q = Query(id=ident or cas or cells.get(ncol or "", "") or f"{source or 'row'}_{n}", cas=cas,
                  name=(cells.get(ncol, "") if ncol else "") or (cells.get(dcol, "") if dcol else ""),
                  smiles=cells.get(scol, "") if scol else "",
                  inchikey=cells.get(ik_drawn, "") if ik_drawn else "",
                  species_inchikey=cells.get(ik_species, "") if ik_species else "", cid=cid, row=cells)
        out.append(q)
    return out


def read_queries(path: str | Path, id_col: str | None = None, smiles_col: str | None = None,
                 sheet=None, log: Callable[[str], None] | None = None) -> list[Query]:
    """Molecules to enrich from a CSV/Excel table (CAS and/or SMILES columns), an SD file, a QuantaBench run
    folder or merged tables (summary.csv with id, input_inchikey, inchikey, input_smiles), or paired.csv."""
    path = Path(path).expanduser()
    if path.is_file() and path.suffix.lower() == ".sdf":
        from .inputs import read_sdf
        from .validate import species_keys

        out = []
        for m in read_sdf(path, id_col=id_col, log=log):
            props = {k: str(v) for k, v in (m.meta or {}).items()}
            cas = pc.normalize_cas(m.id) or next((pc.normalize_cas(props[k]) for k in props
                                                  if k.lower().strip() in CAS_COLUMNS
                                                  and pc.normalize_cas(props[k])), "")
            keys = species_keys(m.mol)
            out.append(Query(id=m.id, cas=cas, name=m.name, smiles=m.smiles, inchikey=keys["inchikey"], mol=m.mol,
                             row={"id": m.id, "name": m.name, **props}))
        return out
    return queries_from_rows(_read_rows(path, sheet), id_col, smiles_col, source=path.stem)


# ============================================================================================ structures
def _parent_key(smiles: str = "", mol=None) -> str:
    """InChIKey of the largest fragment, neutralized: a salt's parent (K-PFOS -> PFOS acid)."""
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _quiet = rdBase.BlockLogs()
    try:
        m = Chem.Mol(mol) if mol is not None else Chem.MolFromSmiles(smiles) if smiles else None
        if m is None:
            return ""
        m = rdMolStandardize.LargestFragmentChooser().choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        return Chem.MolToInchiKey(m) or ""
    except Exception:  # noqa: BLE001 - no key is an answer here
        return ""
    finally:
        del _quiet


def qb_keys(q: Query) -> list[tuple[str, str]]:
    """(label, InChIKey) of QuantaBench's structure for a molecule: as drawn, computed species, parent."""
    from .validate import keys_from_smiles, species_keys

    drawn = q.inchikey
    if not drawn and q.mol is not None:
        drawn = species_keys(q.mol)["inchikey"]
    if not drawn and q.smiles:
        drawn = keys_from_smiles(q.smiles)["inchikey"]
    out = []
    if drawn:
        out.append(("as drawn", drawn))
    if q.species_inchikey and q.species_inchikey != drawn:
        out.append(("computed species", q.species_inchikey))
    parent = _parent_key(q.smiles, q.mol)
    if parent and parent not in {k for _, k in out}:
        out.append(("parent (largest fragment, neutralized)", parent))
    return out


_RANK = {"match": 0, "stereo_differs": 1, "charge_or_protonation_differs": 2, "component_of_mixture": 3,
         "different_structure": 4}


def compare_inchikeys(qb: str, ref: str) -> str:
    """match | stereo_differs | charge_or_protonation_differs | different_structure for two standard InChIKeys."""
    qb, ref = (qb or "").strip(), (ref or "").strip()
    if qb == ref:
        return "match"
    if len(qb) == 27 and len(ref) == 27 and qb[:14] == ref[:14]:
        return "stereo_differs" if qb[-1] == ref[-1] else "charge_or_protonation_differs"
    return "different_structure"


def _components(smiles: str) -> list[tuple[str, int, str]]:
    """(InChIKey, heavy atoms, SMILES) of each component of a multi-component SMILES; [] for one component."""
    if "." not in (smiles or ""):
        return []
    from rdkit import Chem, rdBase

    _quiet = rdBase.BlockLogs()
    out = []
    try:
        for frag in dict.fromkeys(smiles.split(".")):
            m = Chem.MolFromSmiles(frag)
            if m is not None:
                out.append((Chem.MolToInchiKey(m) or "", m.GetNumHeavyAtoms(), frag))
    except Exception:  # noqa: BLE001 - no components is an answer here
        return []
    finally:
        del _quiet
    return out


def compare_structures(keys: list[tuple[str, str]], pubchem_inchikey: str,
                       pubchem_smiles: str = "") -> tuple[str, str]:
    """(outcome, note) of QuantaBench's keys against PubChem's InChIKey: the best agreement over the keys. A match of
    the parent only (a salt drawn, PubChem's CAS record the parent acid, or the reverse) is reported as
    ``charge_or_protonation_differs``. With ``pubchem_smiles`` of a multi-component record, QuantaBench's structure
    equal to one component (same first block) is ``component_of_mixture``."""
    if not keys:
        return "no_structure", "no QuantaBench structure key"
    if not pubchem_inchikey:
        return "error", "PubChem gave no InChIKey"
    best, best_label = "different_structure", ""
    for label, key in keys:
        o = compare_inchikeys(key, pubchem_inchikey)
        if label.startswith("parent") and o == "match":
            o = "charge_or_protonation_differs"
        if _RANK[o] < _RANK[best]:
            best, best_label = o, label
    if best == "different_structure":
        comps = _components(pubchem_smiles)
        hit = next(((label, c) for label, key in keys for c in comps
                    if len(key) == 27 and len(c[0]) == 27 and key[:14] == c[0][:14]), None)
        if hit is not None:
            label, (_ik, heavy, smi) = hit
            others = [c[2] for c in comps if c[0][:14] != hit[1][0][:14]]
            note = f"QuantaBench has {smi} ({label}); PubChem's record also has {'.'.join(others)}"
            if heavy < max(c[1] for c in comps):
                note += "; QuantaBench kept a smaller component than the largest"
            return "component_of_mixture", note
        return best, f"PubChem {pubchem_inchikey} vs QuantaBench {keys[0][1]}"
    note = f"on the {best_label} structure" if best_label != "as drawn" else ""
    if best != "match":
        note = (note + "; " if note else "") + f"PubChem {pubchem_inchikey}"
    return best, note


# ============================================================================================ enrichment
@dataclass
class Resolved:
    status: str = ""                 # ok | ambiguous | not_found | error | not_cached
    query: str = ""
    query_type: str = ""
    cids: list[int] = field(default_factory=list)
    cid: int | None = None
    note: str = ""
    cas_cids: list[int] | None = None    # what the CAS RN gave (None: no CAS RN looked up)


@dataclass
class Enrichment:
    rows: list[dict]
    records: list[dict]
    counts: dict[str, Counter]
    requests: int
    cache_hits: int
    seconds: float
    stopped: str = ""                # why it stopped early ("" = finished)

    def summary(self) -> str:
        lines = [f"{len(self.rows)} molecule(s): " + ", ".join(f"{k} {v}" for k, v in
                                                                self.counts["status"].most_common())]
        if self.counts.get("query_type"):
            lines.append("looked up by: " + ", ".join(f"{k} {v}" for k, v in self.counts["query_type"].most_common()))
        if self.counts.get("match"):
            lines.append("structure check: " + ", ".join(f"{k} {self.counts['match'][k]}" for k in OUTCOMES
                                                         if self.counts["match"][k]))
        if self.counts.get("experimental"):
            e = self.counts["experimental"]
            lines.append(f"experimental: {e['records']} record(s) for {e['molecules']} molecule(s); surface tension "
                         f"for {e['st_molecules']} (pure liquid {e['st_pure']}, aqueous {e['st_aqueous']}, "
                         f"unclassified {e['st_unknown']}; {e['st_review']} to review)")
        lines.append(f"{self.requests} request(s) to PubChem, {self.cache_hits} answered from the cache, "
                     f"{self.seconds:.0f} s")
        if self.stopped:
            lines.append(f"STOPPED EARLY: {self.stopped}")
        return "\n".join(lines)


def estimate_requests(n: int, experimental: bool) -> int:
    """Upper bound of requests for ``n`` molecules with an empty cache (one lookup each, batched lists, one PUG
    View record each with ``experimental``)."""
    batches = -(-n // pc.PROPERTY_BATCH) + 2 * -(-n // pc.LIST_BATCH)
    return n + batches + (n if experimental else 0)


def _resolve(q: Query, client: pc.PubChemClient) -> Resolved:
    """CIDs of a query: CID, CAS RN, InChIKey, SMILES, name (first that PubChem knows). A CAS RN that PubChem does
    not know is remembered (``cas_cids = []``, for the structure check) and the structure is looked up instead."""
    if q.cid:
        return Resolved("ok", f"cid:{q.cid}", "cid", [q.cid], q.cid)
    cas_cids = None
    tried = []
    for kind, value, fn in (("cas", q.cas, client.cids_by_name), ("inchikey", q.inchikey, client.cids_by_inchikey),
                            ("smiles", q.smiles, client.cids_by_smiles),
                            ("name", q.name if not pc.is_cas(q.name) else "", client.cids_by_name),
                            ("name", q.id if q.id != q.name and not pc.is_cas(q.id) else "", client.cids_by_name)):
        if not value:
            continue
        cids = fn(value)
        tried.append(f"{kind}:{value}")
        if kind == "cas":
            cas_cids = cids
        if cids:
            note = f"CAS RN {q.cas} not in PubChem; found by {kind}" if kind != "cas" and q.cas else ""
            return Resolved("ok" if len(cids) == 1 else "ambiguous", f"{kind}:{value}", kind, cids,
                            cids[0] if len(cids) == 1 else None, note, cas_cids)
    return Resolved("not_found", tried[0] if tried else "", tried[0].split(":")[0] if tried else "", [], None,
                    "PubChem knows none of: " + ", ".join(tried) if tried else "no identifier or structure to look up",
                    cas_cids)


def enrich(queries: list[Query], client: pc.PubChemClient, experimental: bool = False, validate: bool = False,
           progress: Callable[[int, int, str], None] | None = None, max_failures: int = 10) -> Enrichment:
    """Look every query up in PubChem (see the module docstring) and build the enriched rows and, with
    ``experimental``, the long-form experimental records. Stops early (``Enrichment.stopped``) after
    ``max_failures`` consecutive network failures; what was fetched is cached, so a rerun resumes."""
    t0 = time.monotonic()
    req0, hit0 = client.requests, client.cache_hits
    resolved: list[Resolved] = []
    stopped = ""
    failures = 0
    for i, q in enumerate(queries, 1):
        if stopped:
            resolved.append(Resolved("not_looked_up", note="stopped early"))
            continue
        try:
            r = _resolve(q, client)
            failures = 0
        except pc.NotCached as exc:
            r = Resolved("not_cached", note=str(exc))
        except pc.PubChemUnavailable as exc:
            failures += 1
            r = Resolved("error", note=str(exc))
            if failures >= max_failures:
                stopped = f"{failures} lookups in a row could not reach PubChem ({exc}); rerun to resume"
        except pc.PubChemFault as exc:
            r = Resolved("error", note=str(exc))
        resolved.append(r)
        if progress:
            progress(i, len(queries), "lookup")

    all_cids = sorted({c for r in resolved for c in r.cids})
    props: dict[int, dict] = {}
    rns: dict[int, list[str]] = {}
    syns: dict[int, list[str]] = {}
    batch_note = ""
    if all_cids and not stopped:
        try:
            rns = client.registry_numbers(all_cids)
            props = client.properties(all_cids)
            chosen = sorted({r.cid for r in resolved if r.cid} | {c for r in resolved if r.status == "ambiguous"
                                                                  for c in r.cids})
            syns = client.synonyms(chosen)
        except pc.PubChemError as exc:
            batch_note = f"batched data incomplete: {exc}"

    rows, records = [], []
    counts: dict[str, Counter] = {"status": Counter(), "query_type": Counter(), "match": Counter(),
                                  "experimental": Counter()}
    for i, (q, r) in enumerate(zip(queries, resolved, strict=True), 1):
        note = [n for n in (r.note, batch_note) if n]
        keys = qb_keys(q) if validate else []
        # a CAS RN with several CIDs: keep those that carry it as a registry number
        if r.query_type == "cas" and len(r.cids) > 1:
            carried = [c for c in r.cids if q.cas in pc.cas_numbers(rns.get(c, []))]
            if len(carried) == 1:
                note.append(f"{len(r.cids)} CIDs for {q.cas}; only CID {carried[0]} lists it as a registry number")
                r.cids, r.cid, r.status = carried, carried[0], "ok"
            elif carried:
                r.cids = carried
        if r.status == "ambiguous" and keys:
            same = [c for c in r.cids if props.get(c, {}).get("InChIKey") in {k for _, k in keys}]
            if len(same) == 1:
                r.cid = same[0]
                note.append(f"several CIDs; CID {r.cid} has QuantaBench's structure")
        cid = r.cid if r.cid is not None else None
        p = props.get(cid, {}) if cid else {}
        cas_all = pc.cas_numbers(syns.get(cid, []) + rns.get(cid, [])) if cid else []
        cas_syn = pc.cas_numbers(syns.get(cid, [])) if cid else []
        out = dict(q.row)
        out.setdefault("id", q.id)
        out.update({"pubchem_status": r.status, "pubchem_query": r.query, "pubchem_query_type": r.query_type,
                    "pubchem_cid": cid or "", "pubchem_cids": ";".join(map(str, r.cids)),
                    "pubchem_cas": cas_syn[0] if cas_syn else (cas_all[0] if cas_all else ""),
                    "pubchem_cas_all": ";".join(cas_all),
                    "pubchem_cas_in_xrefs": ("" if not (q.cas and cid and cid in rns) else
                                             "yes" if q.cas in pc.cas_numbers(rns[cid]) else "no")})
        for col, key in _PROP_MAP:
            out[col] = p.get(key, "")
        if validate:
            out["qb_inchikey"] = keys[0][1] if keys else ""
            if not q.cas:
                outcome, mnote = "no_cas", "no CAS RN to check the structure against"
            elif r.status in ("error", "not_cached", "not_looked_up"):
                outcome, mnote = "error", r.note
            elif r.cas_cids == [] or r.query_type != "cas":
                outcome, mnote = "cas_not_in_pubchem", (f"found by {r.query_type}" if r.cids else "")
            elif len(r.cids) > 1:
                outcome = "ambiguous_cas"
                hits = [c for c in r.cids if compare_structures(keys, props.get(c, {}).get("InChIKey", ""))[0]
                        == "match"]
                mnote = f"CIDs {';'.join(map(str, r.cids))}" + (f"; QuantaBench's structure is CID {hits[0]}"
                                                                 if hits else "")
            else:
                outcome, mnote = compare_structures(keys, p.get("InChIKey", ""), p.get("SMILES", ""))
            out["pubchem_match"], out["pubchem_match_note"] = outcome, mnote
            counts["match"][outcome] += 1
        if experimental:
            recs: list[pc.ExperimentalRecord] = []
            if cid and not stopped:
                try:
                    recs = client.experimental(cid)
                except pc.PubChemError as exc:
                    note.append(f"experimental: {exc}")
                    failures += 1
                    if failures >= max_failures and not isinstance(exc, pc.NotCached):
                        stopped = f"{failures} requests in a row could not reach PubChem ({exc}); rerun to resume"
                else:
                    failures = 0
            out.update(_experimental_summary(recs, p.get("MolecularWeight")))
            e = counts["experimental"]
            e["records"] += len(recs)
            e["molecules"] += bool(recs)
            st = [x for x in recs if x.heading == "Surface Tension" and x.value_std is not None]
            e["st_molecules"] += bool(st)
            e["st_pure"] += any(x.condition_class == "pure_liquid" for x in st)
            e["st_aqueous"] += any(x.condition_class == "aqueous_solution" for x in st)
            e["st_unknown"] += any(x.condition_class == "unknown" for x in st)
            e["st_review"] += any(x.needs_review for x in st)
            for x in recs:
                records.append({"id": q.id, "input_cas": q.cas, "pubchem_inchikey": p.get("InChIKey", ""),
                                **x.row()})
        out["pubchem_note"] = "; ".join(note)
        counts["status"][r.status] += 1
        if r.query_type:
            counts["query_type"][r.query_type] += 1
        rows.append(out)
        if progress and experimental:
            progress(i, len(queries), "experimental")
    return Enrichment(rows, records, counts, client.requests - req0, client.cache_hits - hit0,
                      time.monotonic() - t0, stopped)


def _fmt(v, nd: int = 4):
    if v is None:
        return ""
    return round(v, nd) if isinstance(v, float) else v


def _experimental_summary(recs: list[pc.ExperimentalRecord], mw) -> dict:
    out = {c: "" for c in EXPERIMENTAL_COLUMNS}
    if not recs:
        out["pubchem_exp_records"] = 0
        return out
    st = [r for r in recs if r.heading == "Surface Tension" and r.value_std is not None]
    pure = pc.best_value(recs, "Surface Tension")
    aq = [r.value_std for r in st if r.condition_class == "aqueous_solution"]
    unk = pc.best_value(recs, "Surface Tension", ("unknown",))
    dens = pc.best_value(recs, "Density", ("pure_liquid", "unknown", "solid"),
                         target_t=pure.temperature_c if pure and pure.temperature_c is not None else pc.ROOM_T)
    mp = pc._transition_c(recs, "Melting Point")
    bp = pc._transition_c(recs, "Boiling Point")
    out.update({
        "pubchem_exp_records": len(recs), "pubchem_st_records": len(st),
        "pubchem_st_pure_mN_m": _fmt(pure.value_std if pure else None),
        "pubchem_st_pure_T_C": _fmt(pure.temperature_c if pure else None),
        "pubchem_st_pure_source": f"{pure.source_name}: {pure.reference}"[:300] if pure else "",
        "pubchem_st_aqueous_min_mN_m": _fmt(min(aq) if aq else None),
        "pubchem_st_unclassified_mN_m": _fmt(unk.value_std if unk else None),
        "pubchem_st_review": "yes" if any(r.needs_review for r in st) else "",
        "pubchem_density_g_cm3": _fmt(dens.value_std if dens else None),
        "pubchem_density_T_C": _fmt(dens.temperature_c if dens else None),
        "pubchem_density_class": dens.condition_class if dens else "",
        "pubchem_mp_C": _fmt(mp, 2), "pubchem_bp_C": _fmt(bp, 2),
    })
    if pure and dens and dens.condition_class == "pure_liquid" and (
            pure.temperature_c is None or dens.temperature_c is None
            or abs(pure.temperature_c - dens.temperature_c) <= 10):
        out["pubchem_parachor"] = _fmt(pc.parachor(mw, pure.value_std, dens.value_std), 2)
    return out


def output_columns(rows: list[dict], experimental: bool, validate: bool) -> list[str]:
    """The input's columns in their order, then the pubchem_* columns."""
    ours = list(PUBCHEM_COLUMNS) + (list(VALIDATE_COLUMNS) if validate else []) + \
        (list(EXPERIMENTAL_COLUMNS) if experimental else [])
    seen: list[str] = []
    for r in rows:
        for k in r:
            if k not in ours and k not in seen:
                seen.append(k)
    return seen + ours


def write_csv(path: str | Path, rows: list[dict], columns: Sequence[str] | None = None) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns or (list(rows[0]) if rows else []))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    tmp.replace(path)
    return path
