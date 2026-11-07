"""Structures in and out as text and bytes: the clipboard, drag and drop, and other editors (ChemDraw, Marvin).

In: ``parse_structure_text`` reads a MOL block (V2000 or V3000), SD text (several records), CDXML, Marvin MRV,
or else lines of SMILES, names or CAS numbers (``inputs.parse_text``). ``parse_cdx_bytes`` reads binary
ChemDraw CDX, ChemDraw's own clipboard format. RDKit 2026.09's Python wrapper reads CDX only as CDXML (its
C++ reader handles CDX), so ``cdx_to_cdxml`` translates the CDX objects that describe molecules into CDXML
and RDKit's CDXML reader does the chemistry (nicknames, stereo, charges). A newer RDKit that reads CDX
bytes is used first.

Out: ``to_smiles``, ``to_molblock`` (V3000 only when needed), ``to_sdf`` and ``to_cdxml``, from a molecule, a
``MoleculeInput`` or a ``MoleculeResult``. MOL and CDXML get 2D coordinates (a depiction), since ChemDraw and
other 2D editors read them.

``mdlct_decode`` and ``mdlct_encode`` read and write Windows' "MDLCT" clipboard format (the MOL block's lines,
each after one byte holding its length).

No Qt here: ``gui.clipboard`` reads the clipboard and calls these functions.
"""

from __future__ import annotations

import logging
import re
import struct
from collections.abc import Callable, Iterable, Sequence
from xml.sax.saxutils import escape, quoteattr

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor

from . import inputs
from .inputs import MoleculeInput, safe_id

_log = logging.getLogger(__name__)

CDX_HEADER = b"VjCD0100"
# "VjCD0100", 04 03 02 01 and 10 reserved bytes; the document object's tag and id follow (the SDK's 28-byte
# kCDX_HeaderLength counts them)
_CDX_DOCUMENT_AT = 22
_COORD = 65536.0             # CDX coordinates are 1/65536 point; CDXML uses points

# ================================================================== reading text


def detect_format(text: str) -> str:
    """"cdxml", "mrv", "sdf", "mol" or "text" (lines of SMILES, names, CAS: ``inputs.parse_text``)."""
    head = text.lstrip("\ufeff \t\r\n")[:4096]
    if "<CDXML" in head:
        return "cdxml"
    if "<MDocument" in head or ("<cml" in head and "<molecule" in head):
        return "mrv"
    if re.search(r"^\$\$\$\$\s*$", text, re.M):
        return "sdf"
    if re.search(r"^M  END\s*$", text, re.M):
        return "mol"
    return "text"


def _input_from_mol(mol: Chem.Mol, source: str, number: int = 0) -> MoleculeInput:
    """A ``MoleculeInput`` holding ``mol``, named like ``inputs.read_sdf`` names records (DTXSID or ID property,
    else the title); without either, the SMILES is the id, as for SMILES typed in the Molecules box."""
    title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    props = {k: str(v) for k, v in mol.GetPropsAsDict().items() if not k.startswith("_") and k != "CDX_FRAG_ID"}
    with rdBase.BlockLogs():       # "not removing hydrogen atom without neighbors" for H- fragments
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    ident = next((props[k].strip() for k in inputs.SDF_ID_PROPERTIES if props.get(k, "").strip()), "") or title
    fallback = f"{source}_{number}" if number else source
    return MoleculeInput(id=safe_id(ident or smiles, fallback=safe_id(fallback, "mol")), name=title or ident,
                         source=f"{source}#{number}" if number else source, mol=mol, smiles=smiles, meta=props)


def _merge_ions(mols: Sequence[Chem.Mol]) -> list[Chem.Mol]:
    """ChemDraw keeps each fragment of a drawing apart, so a salt (K+ and PFOS-) comes back as two molecules.
    When the charged fragments' charges add up to zero, they are one salt: join them (neutral fragments stay
    separate molecules). Otherwise the fragments are left as they are."""
    charged = [m for m in mols if Chem.GetFormalCharge(m) != 0]
    if len(charged) < 2 or sum(Chem.GetFormalCharge(m) for m in charged) != 0:
        return list(mols)
    salt = charged[0]
    for m in charged[1:]:
        salt = Chem.CombineMols(salt, m)
    out, placed = [], False
    for m in mols:
        if Chem.GetFormalCharge(m) == 0:
            out.append(m)
        elif not placed:
            out.append(salt)            # where the first ion was
            placed = True
    return out


def _chemdraw_inputs(mols: Sequence[Chem.Mol], source: str) -> list[MoleculeInput]:
    mols = _merge_ions([m for m in mols if m is not None and m.GetNumAtoms()])
    if not mols:
        raise ValueError(f"No molecule could be read from the {source} (ChemDraw) data")
    items = [_input_from_mol(m, source, i if len(mols) > 1 else 0) for i, m in enumerate(mols, 1)]
    return inputs.dedupe_ids(items)


def read_cdxml(text: str, source: str = "CDXML") -> list[MoleculeInput]:
    from rdkit.Chem import rdChemDraw

    try:
        with rdBase.BlockLogs():
            mols = rdChemDraw.MolsFromChemDrawBlock(text)
    except RuntimeError as exc:      # RDKit invariant violations (e.g. a nickname it cannot attach)
        raise ValueError(f"RDKit could not read the {source} (ChemDraw) data: "
                         f"{str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__}") from exc
    return _chemdraw_inputs(mols, source)


def _read_sd_text(text: str, source: str, log: Callable[[str], None] | None) -> list[MoleculeInput]:
    supplier = Chem.SDMolSupplier()
    with rdBase.BlockLogs():
        supplier.SetData(text, removeHs=False)
        mols = list(supplier)
    out, skipped = [], []
    for i, mol in enumerate(mols, 1):
        if mol is None or not mol.GetNumAtoms():
            skipped.append(str(i))
            continue
        out.append(_input_from_mol(mol, source, i if len(mols) > 1 else 0))
    warnings: list[str] = []
    if skipped:
        warnings.append(f"{source}: skipped {len(skipped)} record(s) RDKit cannot read (record {', '.join(skipped)})")
    out = inputs.dedupe_ids(out, warnings)
    for w in warnings:
        (log or _log.warning)(w)
    if not out:
        raise ValueError(f"No molecule could be read from the {source} (SD/MOL text)")
    return out


def parse_structure_text(text: str, source: str = "clipboard",
                         log: Callable[[str], None] | None = None) -> list[MoleculeInput]:
    """Molecules in ``text``, whatever its format (``detect_format``). Lines of SMILES and names go through
    ``inputs.parse_text`` (names and CAS numbers are looked up on PubChem, so call this off the GUI thread);
    a line that cannot be read is reported through ``log`` and skipped. Raises ``ValueError`` when nothing
    could be read."""
    kind = detect_format(text)
    if kind == "cdxml":
        return read_cdxml(text, source)
    if kind == "mrv":
        with rdBase.BlockLogs():
            mol = Chem.MolFromMrvBlock(text, removeHs=False)
        if mol is None:
            raise ValueError(f"No molecule could be read from the {source} (Marvin MRV)")
        return [_input_from_mol(mol, source)]
    if kind in ("sdf", "mol"):
        if kind == "mol" and not text.rstrip().endswith("$$$$"):
            text = text.rstrip("\r\n") + "\n$$$$\n"
        return _read_sd_text(text, source, log)
    return inputs.parse_text(text, skip_errors=True, log=log, source=source)


# ================================================================== MDLCT (Windows clipboard)


def mdlct_decode(data: bytes) -> str:
    """Windows' "MDLCT" clipboard format -> MOL block text. Each line is stored after a byte giving its length,
    with no line ends. Data that does not follow that layout is returned as text, as some programs put a
    plain MOL block there."""
    lines, i = [], 0
    while i < len(data):
        n = data[i]
        if i + 1 + n > len(data):
            break
        lines.append(data[i + 1:i + 1 + n].decode("latin-1"))
        i += 1 + n
    text = "\n".join(lines)
    if i == len(data) and re.search(r"^M  END", text, re.M):
        return text + "\n"
    return data.decode("latin-1").replace("\r\n", "\n").rstrip("\0")


def mdlct_encode(molblock: str) -> bytes:
    out = bytearray()
    for line in molblock.splitlines():
        raw = line.encode("latin-1", "replace")[:255]
        out.append(len(raw))
        out += raw
    return bytes(out)


# ================================================================== binary ChemDraw (CDX)
# Tags, enums and CDXML names are those of the CDX format (the ChemDraw SDK's CDXConstants.h and CDXMLNames,
# BSD-3-Clause, as bundled with RDKit). Only objects and properties that describe molecules are translated.

_CDX_OBJECTS = {0x8000: "CDXML", 0x8001: "page", 0x8002: "group", 0x8003: "fragment", 0x8004: "n",
                0x8005: "b", 0x8006: "t"}

_NODE_TYPES = ("Unspecified", "Element", "ElementList", "ElementListNickname", "Nickname", "Fragment",
               "Formula", "GenericNickname", "AnonymousAlternativeGroup", "NamedAlternativeGroup",
               "MultiAttachment", "VariableAttachment", "ExternalConnectionPoint", "LinkNode", "Monomer")
_RADICALS = ("None", "Singlet", "Doublet", "Triplet")
_GEOMETRIES = ("Unknown", "1", "Linear", "Bent", "TrigonalPlanar", "TrigonalPyramidal", "SquarePlanar",
               "Tetrahedral", "TrigonalBipyramidal", "SquarePyramidal", "5", "Octahedral", "6", "7", "8", "9", "10")
_ATOM_CIP = ("U", "N", "R", "S", "r", "s", "u", "M", "P", "a", "-", "+", "?")
_BOND_CIP = ("U", "N", "E", "Z")
_ENHANCED_STEREO = ("Unspecified", "None", "Absolute", "Or", "And")
_EXTERNAL_CONNECTIONS = ("Unspecified", "Diamond", "Star", "PolymerBead", "Wavy", "Residue", "Peptide", "DNA",
                         "RNA", "Terminus", "Sulfide", "Nucleotide", "UnlinkedBranch", "Chem")
_BOND_DISPLAY = ("Solid", "Dash", "Hash", "WedgedHashBegin", "WedgedHashEnd", "Bold", "WedgeBegin", "WedgeEnd",
                 "Wavy", "HollowWedgeBegin", "HollowWedgeEnd", "WavyWedgeBegin", "WavyWedgeEnd", "Dot", "DashDot",
                 "", "", "", "", "DottedHydrogen")
_BOND_ORDERS = ((0x0001, "1"), (0x0002, "2"), (0x0004, "3"), (0x0008, "4"), (0x0010, "5"), (0x0020, "6"),
                (0x0040, "0.5"), (0x0080, "1.5"), (0x0100, "2.5"), (0x0200, "3.5"), (0x0400, "4.5"),
                (0x0800, "5.5"), (0x1000, "dative"), (0x2000, "ionic"), (0x4000, "hydrogen"),
                (0x8000, "threecenter"))


def _uint(raw: bytes) -> int:
    return int.from_bytes(raw, "little", signed=False) if raw else 0


def _int(raw: bytes) -> int:
    return int.from_bytes(raw, "little", signed=True) if raw else 0


def _enum(names: Sequence[str]) -> Callable[[bytes], str | None]:
    def conv(raw: bytes) -> str | None:
        v = _uint(raw)
        return names[v] if v < len(names) and names[v] else None
    return conv


def _point(raw: bytes) -> str | None:
    if len(raw) < 8:
        return None
    y, x = struct.unpack_from("<ii", raw)       # CDX stores the vertical coordinate first
    return f"{x / _COORD:.4f} {y / _COORD:.4f}"


def _ids(raw: bytes) -> str | None:
    n = len(raw) // 4
    return " ".join(str(v) for v in struct.unpack_from(f"<{n}I", raw)) if n else None


def _cdx_string(raw: bytes) -> str:
    """CDXString: a UINT16 count of 10-byte style runs, the runs, then the text."""
    if len(raw) < 2:
        return ""
    runs = struct.unpack_from("<H", raw)[0]
    return raw[2 + 10 * runs:].decode("cp1252", "replace").replace("\r", "\n")


def _charge(raw: bytes) -> str:
    if len(raw) == 1:
        return str(_int(raw))
    value = _int(raw) / float(1 << 24)        # INT32 charges are fixed point with 24 fraction bits
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _bond_order(raw: bytes) -> str | None:
    v = _uint(raw)
    if v in (0xFF, 0xFFFF) and len(raw) <= 2:
        return "any"
    names = [name for bit, name in _BOND_ORDERS if v & bit]
    return " ".join(names) or None


def _flag(raw: bytes) -> str:          # a flag property: present means yes
    return "yes"


# tag -> (CDXML attribute, value converter). Converters return None to leave the attribute out.
_CDX_PROPS: dict[int, tuple[str, Callable[[bytes], str | None]]] = {
    0x0200: ("p", _point),
    0x0400: ("NodeType", _enum(_NODE_TYPES)),
    0x0402: ("Element", lambda raw: str(_uint(raw))),
    0x0420: ("Isotope", lambda raw: str(_int(raw))),
    0x0421: ("Charge", _charge),
    0x0422: ("Radical", _enum(_RADICALS)),
    0x0429: ("AbnormalValence", _flag),
    0x042B: ("NumHydrogens", lambda raw: str(_uint(raw))),
    0x042E: ("HDot", _flag),
    0x042F: ("HDash", _flag),
    0x0430: ("Geometry", _enum(_GEOMETRIES)),
    0x0431: ("BondOrdering", _ids),
    0x0433: ("GenericNickname", _cdx_string),
    0x0437: ("AS", _enum(_ATOM_CIP)),
    0x0440: ("ExternalConnectionType", _enum(_EXTERNAL_CONNECTIONS)),
    0x0446: ("EnhancedStereoType", _enum(_ENHANCED_STEREO)),
    0x0447: ("EnhancedStereoGroupNum", lambda raw: str(_int(raw))),
    0x0448: ("NeedsClean", _flag),
    0x044B: ("ExternalConnectionNum", lambda raw: str(_int(raw))),
    0x0505: ("ConnectionOrder", _ids),
    0x0600: ("Order", _bond_order),
    0x0601: ("Display", _enum(_BOND_DISPLAY)),
    0x0602: ("Display2", _enum(_BOND_DISPLAY)),
    0x0604: ("B", lambda raw: str(_uint(raw))),
    0x0605: ("E", lambda raw: str(_uint(raw))),
    0x0608: ("BeginAttach", lambda raw: str(_uint(raw))),
    0x0609: ("EndAttach", lambda raw: str(_uint(raw))),
    0x060A: ("BS", _enum(_BOND_CIP)),
    0x0611: ("BeginExternalNum", lambda raw: str(_int(raw))),
    0x0612: ("EndExternalNum", lambda raw: str(_int(raw))),
    0x0805: ("BondLength", lambda raw: f"{_int(raw) / _COORD:g}"),
}
_TEXT = 0x0700


class _CDXReader:
    """Walks the CDX tag stream: an object is a UINT16 tag (high bit set) and a UINT32 id, followed by its
    properties and child objects, and ends with a zero tag. A property is a UINT16 tag, a UINT16 length (0xFFFF:
    a UINT32 length follows) and its data."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = _CDX_DOCUMENT_AT if data.startswith(CDX_HEADER) else 0
        self.out: list[str] = []

    def _read(self, fmt: str):
        values = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return values

    def run(self) -> str:
        self.out.append('<?xml version="1.0" encoding="UTF-8" ?>\n')
        if struct.unpack_from("<H", self.data, self.pos)[0] == 0x8000:
            self.pos += 2
            self._object(0x8000, emit=True)
        else:        # data that starts with the document's own properties (no header, tag or id)
            self._object(0x8000, emit=True, obj_id=0)
        return "".join(self.out)

    def _object(self, tag: int, emit: bool, depth: int = 0, obj_id: int | None = None) -> None:
        if depth > 64:
            raise ValueError("CDX objects nested too deeply")
        if obj_id is None:
            (obj_id,) = self._read("<I")
        name = _CDX_OBJECTS.get(tag) if emit else None
        attrs: list[str] = [f'id="{obj_id}"'] if name else []
        children: list[str] = []
        text = None
        saved, self.out = self.out, children
        try:
            while True:
                if self.pos + 2 > len(self.data):
                    if depth == 0:          # the document's closing tag is missing: its end is the data's end
                        break
                    raise ValueError("CDX data ends inside an object")
                (sub,) = self._read("<H")
                if sub == 0:
                    break
                if sub & 0x8000:
                    self._object(sub, emit=name is not None, depth=depth + 1)
                    continue
                (size,) = self._read("<H")
                if size == 0xFFFF:
                    (size,) = self._read("<I")
                raw = self.data[self.pos:self.pos + size]
                if len(raw) < size:
                    raise ValueError("CDX data ends inside a property")
                self.pos += size
                if not name:
                    continue
                if sub == _TEXT and name == "t":
                    text = _cdx_string(raw)
                    continue
                known = _CDX_PROPS.get(sub)
                if known is not None:
                    value = known[1](raw)
                    if value is not None:
                        attrs.append(f"{known[0]}={quoteattr(value)}")
        finally:
            self.out = saved
        if not name:
            return
        self.out.append(f"<{name} {' '.join(attrs)}>")
        if text is not None:
            self.out.append(f"<s>{escape(text)}</s>")
        self.out.extend(children)
        self.out.append(f"</{name}>")


def cdx_to_cdxml(data: bytes) -> str:
    """Binary ChemDraw (CDX) -> CDXML text with the document's pages, groups, fragments, nodes, bonds and
    text labels (graphics, arrows, reaction schemes and other objects are left out). The file header
    ("VjCD0100...") is optional, as clipboard data may come without it."""
    if len(data) < 2:
        raise ValueError("Not ChemDraw CDX data (empty)")
    try:
        return _CDXReader(bytes(data)).run()
    except struct.error as exc:
        raise ValueError("CDX data ends early") from exc


def _rdkit_reads_cdx(data: bytes) -> tuple[Chem.Mol, ...] | None:
    """Molecules from RDKit's own CDX reader, when this RDKit's Python wrapper offers it (newer than 2026.09:
    a ``format`` argument), else None."""
    from rdkit.Chem import rdChemDraw

    fmt = getattr(rdChemDraw.CDXFormat, "CDX", None)
    if fmt is None:
        return None
    try:
        with rdBase.BlockLogs():
            return tuple(rdChemDraw.MolsFromChemDrawBlock(data, format=fmt))
    except Exception:  # noqa: BLE001 - ArgumentError: this wrapper has no format argument
        return None


def parse_cdx_bytes(data: bytes, source: str = "CDX") -> list[MoleculeInput]:
    """Molecules in binary ChemDraw (CDX) data, such as ChemDraw's clipboard format ("ChemDraw Interchange
    Format" on Windows, ``com.cambridgesoft.chemdraw.cdx`` on macOS) or a ``.cdx`` file's bytes."""
    data = bytes(data)
    if data.lstrip()[:1] == b"<":                # CDXML under a CDX name
        return read_cdxml(data.decode("utf-8", "replace"), source)
    mols = _rdkit_reads_cdx(data) if data.startswith(CDX_HEADER) else None
    if mols:
        return _chemdraw_inputs(mols, source)
    try:
        xml = cdx_to_cdxml(data)
    except ValueError as exc:
        raise ValueError(f"The {source} data is not ChemDraw CDX that QuantaBench can read: {exc}") from exc
    return read_cdxml(xml, source)


# ================================================================== writing


def _mol_of(obj) -> tuple[Chem.Mol, str]:
    """(copy of the molecule, its name) of an RDKit molecule, a ``MoleculeInput`` or a ``MoleculeResult``."""
    if isinstance(obj, Chem.Mol):
        return Chem.Mol(obj), (obj.GetProp("_Name") if obj.HasProp("_Name") else "")
    if isinstance(obj, MoleculeInput):
        return obj.to_mol(), obj.name or obj.id
    mol, inp = getattr(obj, "mol", None), getattr(obj, "input", None)
    name = (inp.name or inp.id) if isinstance(inp, MoleculeInput) else ""
    if mol is not None:
        return Chem.Mol(mol), name
    if isinstance(inp, MoleculeInput):
        return inp.to_mol(), name
    raise TypeError(f"Not a molecule: {type(obj).__name__}")


def _plain(obj) -> tuple[Chem.Mol, str]:
    """Hydrogens removed (those that define stereo stay), stereo taken from 3D coordinates when there are some."""
    mol, name = _mol_of(obj)
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        Chem.AssignStereochemistryFrom3D(mol)
    with rdBase.BlockLogs():
        try:
            return Chem.RemoveHs(mol), name
        except Exception:  # noqa: BLE001 - a structure RDKit cannot sanitize: write it as it is
            mol = Chem.RemoveHs(mol, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            return mol, name


def _depiction(obj) -> tuple[Chem.Mol, str]:
    """The molecule with 2D coordinates (its own when they are 2D, else computed)."""
    mol, name = _plain(obj)
    if not mol.GetNumConformers() or mol.GetConformer().Is3D():
        rdDepictor.Compute2DCoords(mol)
    if name:
        mol.SetProp("_Name", name)
    return mol, name


def to_smiles(obj) -> str:
    mol, _ = _plain(obj)
    return Chem.MolToSmiles(mol)


def _needs_v3000(mol: Chem.Mol) -> bool:
    return mol.GetNumAtoms() > 999 or mol.GetNumBonds() > 999 or bool(mol.GetStereoGroups())


def to_molblock(obj) -> str:
    """MOL block with 2D coordinates: V2000, or V3000 when V2000 cannot hold the molecule (more than 999 atoms
    or bonds, enhanced stereo groups)."""
    mol, _ = _depiction(obj)
    return Chem.MolToMolBlock(mol, forceV3000=_needs_v3000(mol))


def to_sdf(objs: Iterable) -> str:
    """SD text, one record per molecule (2D coordinates, as ``to_molblock``)."""
    return "".join(to_molblock(o) + "$$$$\n" for o in objs)


def to_cdxml(objs) -> str:
    """CDXML (ChemDraw's XML) of one molecule or of several (laid out side by side in one drawing)."""
    from rdkit.Chem import rdChemDraw

    items = list(objs) if isinstance(objs, (list, tuple)) else [objs]
    if not items:
        raise ValueError("No molecule to write")
    if len(items) == 1:
        mol, _ = _depiction(items[0])
    else:
        mol = None
        for o in items:
            m, _ = _plain(o)
            mol = m if mol is None else Chem.CombineMols(mol, m)
        rdDepictor.Compute2DCoords(mol)
    return rdChemDraw.MolToChemDrawBlock(mol)
