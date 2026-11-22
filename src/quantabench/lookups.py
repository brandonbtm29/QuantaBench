"""Web lookups (SETTINGS-1; Spartan's Preferences > URLs): URL templates that the molecule list's *Look up* menu opens
in the browser, with the molecule's identifiers in place of the placeholders.

    %inchikey   standard InChIKey of the structure as drawn (all fragments, charges as drawn: a salt's record)
    %inchi      standard InChI of the same
    %cas        a CAS RN: a posted data field named like CAS (CAS, CASRN, CAS_RN, ...), else any posted field, the id or
                the name that is a valid CAS RN (check digit included)
    %smiles     the input SMILES (canonical, from the structure when the input has none)
    %name       the name (the id when there is none)

Each value is percent-encoded for a URL. A template that needs a value the molecule does not have (no CAS RN) is
offered greyed out, with the reason. Templates live in ``settings.lookups.urls`` ({"name", "url"}); the defaults are
``settings.DEFAULT_LOOKUP_URLS``.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: the placeholders, longest first (``%inchikey`` before ``%inchi``)
PLACEHOLDERS = ("%inchikey", "%inchi", "%smiles", "%name", "%cas")
#: what each placeholder stands for, for the settings tab
PLACEHOLDER_HELP = {"%inchikey": "InChIKey", "%inchi": "InChI", "%cas": "CAS RN", "%smiles": "SMILES",
                    "%name": "name"}
_TOKEN = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS))
_CAS_FIELD = re.compile(r"\bcas\b|^cas|cas_?rn|casrn", re.IGNORECASE)


@dataclass(frozen=True)
class Lookup:
    name: str
    url: str


def check(url: str) -> str:
    """Why ``url`` is not a usable template ("" when it is): an http(s) address."""
    text = str(url or "").strip()
    if not text:
        return "no URL"
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return "not an http:// or https:// address"
    return ""


def placeholders(url: str) -> list[str]:
    """The placeholders ``url`` uses, in order of first use."""
    out: list[str] = []
    for m in _TOKEN.finditer(url or ""):
        if m.group(0) not in out:
            out.append(m.group(0))
    return out


def templates(entries: Iterable | None = None) -> list[Lookup]:
    """The usable templates of ``entries`` (``settings.lookups.urls``; default: the saved settings'): those with a
    name and a valid URL, in order."""
    if entries is None:
        from . import settings

        entries = settings.load().lookups.urls
    out = []
    for e in entries or ():
        if not isinstance(e, Mapping):
            continue
        name, url = str(e.get("name") or "").strip(), str(e.get("url") or "").strip()
        if name and not check(url):
            out.append(Lookup(name, url))
    return out


def expand(url: str, values: Mapping[str, str]) -> tuple[str, list[str]]:
    """(``url`` with each placeholder replaced by its percent-encoded value, the placeholders that have no value)."""
    missing = [p for p in placeholders(url) if not str(values.get(p[1:]) or "").strip()]

    def repl(m: re.Match) -> str:
        return urllib.parse.quote(str(values.get(m.group(0)[1:]) or "").strip(), safe="")

    return _TOKEN.sub(repl, url), missing


def find_cas(fields: Mapping | None = None, *candidates: str) -> str:
    """A CAS RN for a molecule: posted fields named like CAS first, then any field, then ``candidates`` (id, name)."""
    from .chem.pubchem import cas_numbers, normalize_cas

    fields = fields or {}
    for key, value in fields.items():
        cas = normalize_cas(str(value)) if _CAS_FIELD.search(str(key)) else ""
        if cas:
            return cas
    found = cas_numbers([str(v) for v in [*fields.values(), *candidates] if v is not None])
    return found[0] if found else ""


def values_for(mol=None, smiles: str = "", name: str = "", ident: str = "", fields: Mapping | None = None) -> dict:
    """The placeholder values of one molecule: ``mol`` (an RDKit molecule as drawn) or ``smiles`` gives the structure
    keys; ``fields`` (posted data fields), ``ident`` and ``name`` give the CAS RN."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        if mol is None and smiles:
            mol = Chem.MolFromSmiles(smiles)
        inchi = key = ""
        if mol is not None:
            from rdkit.Chem import inchi as rd_inchi

            if rd_inchi.INCHI_AVAILABLE:
                try:
                    inchi = rd_inchi.MolToInchi(mol) or ""
                    key = rd_inchi.InchiToInchiKey(inchi) if inchi else ""
                except Exception:  # noqa: BLE001 - InChI cannot represent it: no key
                    inchi = key = ""
            smiles = smiles or Chem.MolToSmiles(mol)
    return {"inchikey": key, "inchi": inchi, "smiles": smiles or "", "name": name or ident or "",
            "cas": find_cas(fields, ident, name)}
