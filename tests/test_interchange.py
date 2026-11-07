"""chem.interchange: structures from clipboard text and ChemDraw bytes, and SMILES / MOL / CDXML out.

The CDX fixtures are written here by hand (``_cdx``), laid out the way ChemDraw writes them (header, document
properties, a page with fragments, nodes with text labels, bonds, a graphic object to skip), since RDKit
2026.09 cannot write CDX bytes from Python. The reader was also checked against ChemDraw-written CDX files from
RDKit's test data (not committed here).
"""

import struct

import pytest
from conftest import PFOA
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

from quantabench.chem import interchange as ix
from quantabench.chem.inputs import MoleculeInput, from_smiles

GENX = "OC(=O)[C@@](F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F"     # HFPO-DA, one stereocenter
PFOS_ANION = "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


def canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# ------------------------------------------------------------------ a small CDX writer for the fixtures
class _CDX:
    """Writes CDX the way ChemDraw lays it out: tag/length/data properties, objects closed by a zero tag."""

    def __init__(self):
        self.next_id = 100

    def oid(self) -> int:
        self.next_id += 1
        return self.next_id

    @staticmethod
    def prop(tag: int, data: bytes) -> bytes:
        if len(data) >= 0xFFFF:
            return struct.pack("<HHI", tag, 0xFFFF, len(data)) + data
        return struct.pack("<HH", tag, len(data)) + data

    @staticmethod
    def obj(tag: int, oid: int, *parts: bytes) -> bytes:
        return struct.pack("<HI", tag, oid) + b"".join(parts) + b"\0\0"

    @staticmethod
    def string(text: str) -> bytes:   # CDXString: one style run (start, font, face, size, color), then the text
        return struct.pack("<H", 1) + struct.pack("<HHHHH", 0, 3, 96, 200, 0) + text.encode("ascii")

    @staticmethod
    def point(x: float, y: float) -> bytes:   # vertical first, in 1/65536 pt, y down
        return struct.pack("<ii", int(round(-y * 65536)), int(round(x * 65536)))

    def fragment(self, mol: Chem.Mol, scale: float = 30.0, dx: float = 0.0,
                 nickname: dict | None = None) -> bytes:
        """One fragment of ``mol`` (2D coordinates computed, bonds wedged from its stereo). ``nickname`` maps an
        atom index to (label, SMILES of the group with a dummy atom [*] at its attachment) to write that atom as
        a ChemDraw "Fragment" node (a label such as CF3) holding the expanded group."""
        mol = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol)
        Chem.WedgeMolBonds(mol, mol.GetConformer())
        Chem.Kekulize(mol, clearAromaticFlags=True)
        conf = mol.GetConformer()
        ids = {a.GetIdx(): self.oid() for a in mol.GetAtoms()}
        bond_ids = {b.GetIdx(): self.oid() for b in mol.GetBonds()}
        body = []
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            x, y = p.x * scale + dx, p.y * scale
            parts = [self.prop(0x0200, self.point(x, y)), self.prop(0x0A00, struct.pack("<H", 7))]   # + a Z order
            children = []
            if nickname and a.GetIdx() in nickname:
                label, group = nickname[a.GetIdx()]
                parts += [self.prop(0x0400, struct.pack("<h", 5)),
                          self.prop(0x0431, b"".join(struct.pack("<I", bond_ids[b.GetIdx()]) for b in a.GetBonds()))]
                children.append(self._group_fragment(group, x, y, scale))
                children.append(self.obj(0x8006, self.oid(), self.prop(0x0700, self.string(label))))
            elif a.GetAtomicNum() != 6 or a.GetFormalCharge():
                parts.append(self.prop(0x0402, struct.pack("<h", a.GetAtomicNum())))
                if a.GetFormalCharge():
                    parts.append(self.prop(0x0421, struct.pack("<b", a.GetFormalCharge())))
                if a.GetTotalNumHs():
                    parts.append(self.prop(0x042B, struct.pack("<H", a.GetTotalNumHs())))
                label = a.GetSymbol() + ("H" * (a.GetTotalNumHs() > 0)) + (
                    {1: "+", -1: "-"}.get(a.GetFormalCharge(), ""))
                children.append(self.obj(0x8006, self.oid(), self.prop(0x0200, self.point(x, y)),
                                         self.prop(0x0700, self.string(label))))
            body.append(self.obj(0x8004, ids[a.GetIdx()], *parts, *children))
        for b in mol.GetBonds():
            order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 4}[b.GetBondType()]
            parts = [self.prop(0x0604, struct.pack("<I", ids[b.GetBeginAtomIdx()])),
                     self.prop(0x0605, struct.pack("<I", ids[b.GetEndAtomIdx()]))]
            if order != 1:
                parts.append(self.prop(0x0600, struct.pack("<H", order)))
            display = {Chem.BondDir.BEGINWEDGE: 6, Chem.BondDir.BEGINDASH: 3}.get(b.GetBondDir())
            if display:
                parts.append(self.prop(0x0601, struct.pack("<h", display)))
            body.append(self.obj(0x8005, bond_ids[b.GetIdx()], *parts))
        return self.obj(0x8003, self.oid(), *body)

    def _group_fragment(self, group_smiles: str, x0: float, y0: float, scale: float) -> bytes:
        """The fragment inside a nickname node: its atoms, and an external connection point for the [*]."""
        g = Chem.MolFromSmiles(group_smiles)
        rdDepictor.Compute2DCoords(g)
        conf = g.GetConformer()
        ids = {a.GetIdx(): self.oid() for a in g.GetAtoms()}
        body = []
        for a in g.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            parts = [self.prop(0x0200, self.point(x0 + p.x * scale / 3, y0 + p.y * scale / 3))]
            if a.GetAtomicNum() == 0:
                parts.append(self.prop(0x0400, struct.pack("<h", 12)))         # ExternalConnectionPoint
            elif a.GetAtomicNum() != 6:
                parts.append(self.prop(0x0402, struct.pack("<h", a.GetAtomicNum())))
            body.append(self.obj(0x8004, ids[a.GetIdx()], *parts))
        for b in g.GetBonds():
            body.append(self.obj(0x8005, self.oid(), self.prop(0x0604, struct.pack("<I", ids[b.GetBeginAtomIdx()])),
                                 self.prop(0x0605, struct.pack("<I", ids[b.GetEndAtomIdx()]))))
        return self.obj(0x8003, self.oid(), *body)

    def document(self, *fragments: bytes) -> bytes:
        graphic = self.obj(0x8007, self.oid(), self.prop(0x0200, self.point(1, 2)),
                           self.obj(0x8006, self.oid(), self.prop(0x0700, self.string("a caption"))))
        page = self.obj(0x8001, self.oid(), self.prop(0x0204, b"\0" * 16), *fragments, graphic)
        doc = self.obj(0x8000, 0, self.prop(0x0003, self.string("hand-written test fixture")),
                       self.prop(0x4000, b"\x01" * 40),                  # a user-defined (unknown) property
                       self.prop(0x0100, b"\x07" * 70000),               # long property: 0xFFFF + UINT32 length
                       self.prop(0x0805, struct.pack("<i", 30 * 65536)), page)
        return b"VjCD0100" + b"\x04\x03\x02\x01" + bytes(10) + doc


def _cdx_of(*smiles: str) -> bytes:
    w = _CDX()
    return w.document(*(w.fragment(Chem.MolFromSmiles(s), dx=300.0 * i) for i, s in enumerate(smiles)))


# ------------------------------------------------------------------ CDX


def test_cdx_bytes_round_trip_pfoa():
    data = _cdx_of(PFOA)
    assert data.startswith(ix.CDX_HEADER)
    items = ix.parse_cdx_bytes(data)
    assert [canon(i.smiles) for i in items] == [canon(PFOA)]
    assert isinstance(items[0], MoleculeInput) and items[0].mol.GetNumConformers() == 1
    assert not items[0].mol.GetConformer().Is3D()          # a 2D drawing: the pipeline embeds it


def test_cdx_keeps_stereo_from_wedges():
    [item] = ix.parse_cdx_bytes(_cdx_of(GENX))
    assert canon(item.smiles) == canon(GENX)
    mirror = GENX.replace("@@", "@")
    assert canon(item.smiles) != canon(mirror)


def test_cdx_salt_fragments_join_and_neutral_ones_stay_apart():
    salt = ix.parse_cdx_bytes(_cdx_of("[K+]", PFOS_ANION))
    assert [canon(i.smiles) for i in salt] == [canon("[K+]." + PFOS_ANION)]
    two = ix.parse_cdx_bytes(_cdx_of(PFOA, "OCCO"))
    assert sorted(canon(i.smiles) for i in two) == sorted([canon(PFOA), canon("OCCO")])
    assert len({i.id for i in two}) == 2


def test_cdx_nickname_node_is_expanded():
    """A CF3 label (a ChemDraw "Fragment" node with the group inside) becomes its atoms: TFA."""
    w = _CDX()
    acetic = Chem.MolFromSmiles("CC(=O)O")
    data = w.document(w.fragment(acetic, nickname={0: ("CF3", "[*]C(F)(F)F")}))
    assert "NodeType=\"Fragment\"" in ix.cdx_to_cdxml(data)
    [item] = ix.parse_cdx_bytes(data)
    assert canon(item.smiles) == canon("OC(=O)C(F)(F)F")


def test_cdx_to_cdxml_translates_molecule_objects_only():
    xml = ix.cdx_to_cdxml(_cdx_of("CC[O-]"))
    assert xml.startswith("<?xml") and "<CDXML" in xml and "<page" in xml and "<fragment" in xml
    assert 'Element="8"' in xml and 'Charge="-1"' in xml and "<s>O-</s>" in xml
    assert "a caption" not in xml                     # the graphic object (and its text) is left out
    assert 'BondLength="30"' in xml


def test_cdx_without_the_file_header():
    """Clipboard data may come without the 22-byte file header, or without the document's tag and id too."""
    data = _cdx_of(PFOA)
    for headless in (data[22:], data[28:], data[28:-2]):
        assert [canon(i.smiles) for i in ix.parse_cdx_bytes(headless)] == [canon(PFOA)]
    cdxml = ix.to_cdxml(Chem.MolFromSmiles(PFOA)).encode()      # CDXML under a CDX name is read as CDXML
    assert [canon(i.smiles) for i in ix.parse_cdx_bytes(cdxml)] == [canon(PFOA)]


def test_cdx_errors():
    with pytest.raises(ValueError):
        ix.parse_cdx_bytes(b"<?xml version='1.0'?><CDXML/>")
    with pytest.raises(ValueError):
        ix.parse_cdx_bytes(bytes(range(7, 200)))
    data = _cdx_of(PFOA)
    with pytest.raises(ValueError, match="ends"):
        ix.cdx_to_cdxml(data[: len(data) // 2])


def test_rdkit_cdx_writer_gap():
    """RDKit 2026.09's Python MolToChemDrawBlock(..., CDXFormat.CDX) cannot return CDX's bytes (it decodes them
    as UTF-8 and fails), which is why the fixtures above are written by hand. When a newer RDKit can, its CDX
    is read here too."""
    from rdkit.Chem import rdChemDraw

    m = Chem.MolFromSmiles(PFOA)
    rdDepictor.Compute2DCoords(m)
    try:
        data = rdChemDraw.MolToChemDrawBlock(m, rdChemDraw.CDXFormat.CDX)
    except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
        pytest.skip(f"this RDKit cannot write CDX bytes from Python ({type(exc).__name__})")
    data = data.encode("latin-1") if isinstance(data, str) else bytes(data)
    assert [canon(i.smiles) for i in ix.parse_cdx_bytes(data)] == [canon(PFOA)]


# ------------------------------------------------------------------ text formats


def _with_name(smiles: str, name: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m)
    m.SetProp("_Name", name)
    return m


def test_detect_format():
    m = _with_name(PFOA, "PFOA")
    assert ix.detect_format(Chem.MolToMolBlock(m)) == "mol"
    assert ix.detect_format(Chem.MolToV3KMolBlock(m)) == "mol"
    assert ix.detect_format(Chem.MolToMolBlock(m) + "$$$$\n") == "sdf"
    assert ix.detect_format(ix.to_cdxml(m)) == "cdxml"
    assert ix.detect_format(Chem.MolToMrvBlock(m)) == "mrv"
    assert ix.detect_format("CCO ethanol\nc1ccccc1") == "text"


def test_mol_blocks_v2000_and_v3000():
    m = _with_name(GENX, "GenX")
    for block in (Chem.MolToMolBlock(m), Chem.MolToV3KMolBlock(m)):
        [item] = ix.parse_structure_text(block)
        assert canon(item.smiles) == canon(GENX)
        assert item.id == "GenX" and item.name == "GenX" and item.mol is not None


def test_sd_text_with_several_records_and_ids():
    a, b = _with_name(PFOA, "PFOA"), _with_name("OCCO", "glycol")
    b.SetProp("DTXSID", "DTXSID8020597")
    sdf = "".join(Chem.MolToMolBlock(x) + "\n".join(f">  <{k}>\n{x.GetProp(k)}\n" for k in x.GetPropNames())
                  + "\n$$$$\n" for x in (a, b))
    items = ix.parse_structure_text(sdf)
    assert [i.id for i in items] == ["PFOA", "DTXSID8020597"]
    assert [canon(i.smiles) for i in items] == [canon(PFOA), canon("OCCO")]
    assert items[1].meta["DTXSID"] == "DTXSID8020597"


def test_cdxml_and_marvin_text():
    m = _with_name(PFOA, "PFOA")
    [c] = ix.parse_structure_text(ix.to_cdxml(m))
    assert canon(c.smiles) == canon(PFOA)
    [r] = ix.parse_structure_text(Chem.MolToMrvBlock(m))
    assert canon(r.smiles) == canon(PFOA)


def test_smiles_lines_go_through_parse_text():
    items = ix.parse_structure_text("OCCO glycol\n\n" + PFOA + "\n")
    assert [i.id for i in items] == ["glycol", ix.safe_id(PFOA)]
    logged = []
    items = ix.parse_structure_text("OCCO\nC1CC(\n", log=logged.append)     # a bad line is skipped and reported
    assert len(items) == 1 and logged and "line 2" in logged[0]
    with pytest.raises(ValueError):
        ix.parse_structure_text("C1CC(")


def test_unreadable_blocks_raise():
    with pytest.raises(ValueError):
        ix.parse_structure_text('<?xml version="1.0"?><CDXML><page><fragment></fragment></page></CDXML>')
    with pytest.raises(ValueError):
        ix.parse_structure_text("junk\n  nothing\n\n  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n")


def test_mdlct_round_trip():
    block = ix.to_molblock(_with_name(PFOA, "PFOA"))
    raw = ix.mdlct_encode(block)
    assert raw[0] == len(block.splitlines()[0])
    assert ix.mdlct_decode(raw).splitlines() == block.splitlines()
    assert ix.mdlct_decode(block.encode()).splitlines() == block.splitlines()      # plain text in MDLCT
    [item] = ix.parse_structure_text(ix.mdlct_decode(raw))
    assert canon(item.smiles) == canon(PFOA)


# ------------------------------------------------------------------ writing


def test_exporters_from_mol_input_and_result():
    from quantabench.chem.pipeline import MoleculeResult

    inp = from_smiles(GENX, id="GenX")
    built = Chem.AddHs(Chem.MolFromSmiles(GENX))
    AllChem.EmbedMolecule(built, randomSeed=7)
    res = MoleculeResult(input=inp, mol=built)
    for obj in (Chem.MolFromSmiles(GENX), inp, res):
        assert canon(ix.to_smiles(obj)) == canon(GENX)
        block = ix.to_molblock(obj)
        assert "V2000" in block.splitlines()[3]
        back = Chem.MolFromMolBlock(block)
        assert not back.GetConformer().Is3D()                 # 2D for ChemDraw, even from a 3D result
        assert canon(Chem.MolToSmiles(back)) == canon(GENX)   # stereo kept (wedges)
        [c] = ix.parse_structure_text(ix.to_cdxml(obj))
        assert canon(c.smiles) == canon(GENX)
    assert ix.to_molblock(inp).splitlines()[0] == "GenX"


def test_molblock_v3000_when_needed_and_sdf():
    big = Chem.MolFromSmiles("C" * 1100)
    assert "V3000" in ix.to_molblock(big).splitlines()[3]
    sdf = ix.to_sdf([from_smiles(PFOA, id="PFOA"), from_smiles("OCCO", id="glycol")])
    assert sdf.count("$$$$") == 2
    assert [i.id for i in ix.parse_structure_text(sdf)] == ["PFOA", "glycol"]


def test_cdxml_of_several_molecules():
    xml = ix.to_cdxml([from_smiles(PFOA, id="PFOA"), from_smiles("OCCO", id="glycol")])
    smiles = sorted(canon(s) for i in ix.parse_structure_text(xml) for s in i.smiles.split("."))
    assert smiles == sorted([canon(PFOA), canon("OCCO")])
    with pytest.raises(ValueError):
        ix.to_cdxml([])
