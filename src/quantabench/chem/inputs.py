"""Turn what the user gives us (SMILES, a name, files, a CSV list) into ``MoleculeInput`` records."""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

SMILES_COLUMNS = ("smiles", "qsar_ready_smiles", "ms_ready_smiles", "canonical_smiles", "isomeric_smiles")
ID_COLUMNS = ("dtxsid", "id", "name", "compound", "preferred_name", "cas", "casrn")
FILE_TYPES = {".sdf", ".mol", ".xyz", ".pdb", ".smi", ".smiles", ".csv", ".txt"}


@dataclass
class MoleculeInput:
    id: str
    smiles: str = ""
    name: str = ""
    source: str = ""
    mol: Chem.Mol | None = None      # set when read from a structure file
    charge: int | None = None        # None = take from the structure after protonation
    multiplicity: int = 1
    meta: dict = field(default_factory=dict)

    def to_mol(self) -> Chem.Mol:
        if self.mol is not None:
            return Chem.Mol(self.mol)
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for {self.id}: {self.smiles!r}")
        return mol


def safe_id(text: str, fallback: str = "mol") -> str:
    """Folder- and file-safe identifier."""
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", text.strip()).strip("._")
    return (s or fallback)[:80]


def looks_like_smiles(text: str) -> bool:
    text = text.strip()
    if not text or " " in text:
        return False
    if any(ch in text for ch in "=#()[]@+\\/") or text.isupper() and len(text) <= 3:
        return Chem.MolFromSmiles(text, sanitize=False) is not None
    # plain words such as "benzene" are names, not SMILES
    return Chem.MolFromSmiles(text) is not None and not text.isalpha()


def from_smiles(smiles: str, id: str | None = None, name: str = "") -> MoleculeInput:
    smiles = smiles.strip()
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Not a valid SMILES string: {smiles!r}")
    return MoleculeInput(id=safe_id(id or name or smiles), smiles=smiles, name=name or id or "", source="smiles")


def lookup_name(name: str, timeout: float = 15.0) -> MoleculeInput:
    """Resolve a chemical name, CAS number or DTXSID-free identifier through PubChem."""
    quoted = urllib.parse.quote(name.strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quoted}/property/SMILES,IsomericSMILES,Title/JSON"
    req = urllib.request.Request(url, headers={"User-Agent": "QuantaBench"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception as exc:
        raise ValueError(f"PubChem lookup failed for {name!r}: {exc}") from exc
    props = (data.get("PropertyTable", {}).get("Properties") or [{}])[0]
    smiles = props.get("SMILES") or props.get("IsomericSMILES") or ""
    if not smiles:
        raise ValueError(f"PubChem returned no structure for {name!r}")
    title = props.get("Title") or name
    return MoleculeInput(id=safe_id(name), smiles=smiles, name=title, source="pubchem",
                         meta={"pubchem_cid": props.get("CID")})


def parse_text(text: str) -> list[MoleculeInput]:
    """Each non-empty line is a SMILES (optionally followed by an id) or a name to look up."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if looks_like_smiles(first):
            out.append(from_smiles(first, id=rest.strip() or None, name=rest.strip()))
        else:
            out.append(lookup_name(line))
    return out


def _pick(columns: list[str], wanted: tuple[str, ...], explicit: str | None) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column {explicit!r} not in file (columns: {', '.join(columns)})")
        return explicit
    lower = {c.lower().strip(): c for c in columns}
    for w in wanted:
        if w in lower:
            return lower[w]
    return None


def read_csv(path: Path, smiles_col: str | None = None, id_col: str | None = None) -> list[MoleculeInput]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        scol = _pick(columns, SMILES_COLUMNS, smiles_col)
        if scol is None:
            raise ValueError(f"No SMILES column found in {path.name} (columns: {', '.join(columns)})")
        icol = _pick(columns, ID_COLUMNS, id_col)
        out = []
        for i, row in enumerate(reader, 1):
            smi = (row.get(scol) or "").strip()
            ident = (row.get(icol) or "").strip() if icol else ""
            if not smi:
                continue
            out.append(MoleculeInput(id=safe_id(ident or f"{path.stem}_{i}"), smiles=smi, name=ident,
                                     source=f"{path.name}:{i + 1}", meta={k: v for k, v in row.items() if k}))
        return out


def read_file(path: str | Path, smiles_col: str | None = None, id_col: str | None = None,
              charge: int = 0) -> list[MoleculeInput]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv(path, smiles_col, id_col)
    if ext in (".smi", ".smiles", ".txt"):
        return parse_text(path.read_text(encoding="utf-8"))
    if ext == ".sdf":
        out = []
        for i, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            ident = (props.get("DTXSID") or props.get("ID") or (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
                     or f"{path.stem}_{i}")
            out.append(MoleculeInput(id=safe_id(str(ident)), name=str(ident), source=f"{path.name}#{i}", mol=mol,
                                     smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
                                     meta={k: str(v) for k, v in props.items()}))
        return out
    if ext == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    elif ext == ".xyz":
        mol = Chem.MolFromXYZFile(str(path))
        if mol is not None:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
    else:
        raise ValueError(f"Unsupported file type {ext} (supported: {', '.join(sorted(FILE_TYPES))})")
    if mol is None:
        raise ValueError(f"Could not read a molecule from {path.name}")
    return [MoleculeInput(id=safe_id(path.stem), name=path.stem, source=path.name, mol=mol,
                          smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)))]
