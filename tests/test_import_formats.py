"""Structure formats of File > Open and ``--file`` (UI-4): each fixture is made here from RDKit molecules (or written
by hand) and must read back to the same canonical SMILES. Open Babel is faked by a small script."""

from __future__ import annotations

import io
import os
import struct
import sys

import pytest
from conftest import PFOA

from quantabench.chem import formats, inputs

TFA = "OC(=O)C(F)(F)F"
ETHANOL = "CCO"
PHENOL = "Oc1ccccc1"


def canon(smi: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def mol3d(smi: str, seed: int = 7):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def smiles_of(items) -> list[str]:
    return [canon(i.smiles) for i in items]


# --------------------------------------------------------------------------------------------- writers for fixtures
def mol2_block(mol, title: str) -> str:
    """A Tripos MOL2 block with SYBYL types for the small molecules used here."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    Chem.Kekulize(m, clearAromaticFlags=False)
    conf = m.GetConformer()
    lines = ["@<TRIPOS>MOLECULE", title, f" {m.GetNumAtoms()} {m.GetNumBonds()} 0 0 0", "SMALL", "NO_CHARGES", "",
             "@<TRIPOS>ATOM"]
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        if sym == "C":
            kind = "C.ar" if a.GetIsAromatic() else {Chem.HybridizationType.SP2: "C.2",
                                                     Chem.HybridizationType.SP: "C.1"}.get(a.GetHybridization(), "C.3")
        elif sym == "O":
            kind = "O.2" if any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()) else "O.3"
        else:
            kind = sym
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetIdx() + 1:7d} {sym}{a.GetIdx() + 1:<6d} {p.x:10.4f} {p.y:10.4f} {p.z:10.4f} {kind:<6s} "
                     "1 MOL 0.0000")
    lines.append("@<TRIPOS>BOND")
    for k, b in enumerate(m.GetBonds(), 1):
        kind = "ar" if b.GetIsAromatic() else {Chem.BondType.DOUBLE: "2", Chem.BondType.TRIPLE: "3"}.get(
            b.GetBondType(), "1")
        lines.append(f"{k:6d} {b.GetBeginAtomIdx() + 1:5d} {b.GetEndAtomIdx() + 1:5d} {kind}")
    return "\n".join(lines) + "\n"


def cdx_bytes(smi: str) -> bytes:
    """A minimal binary ChemDraw CDX document (header, document, page, fragment, nodes, bonds) of ``smi``."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    out = bytearray(b"VjCD0100" + bytes([4, 3, 2, 1]) + bytes(16))
    next_id = [1]

    def obj(tag: int) -> int:
        out.extend(struct.pack("<HI", tag, next_id[0]))
        next_id[0] += 1
        return next_id[0] - 1

    def prop(tag: int, data: bytes) -> None:
        out.extend(struct.pack("<HH", tag, len(data)) + data)

    obj(0x8000)
    prop(0x0003, b"a property of no interest")      # skipped
    obj(0x8001)
    obj(0x8003)
    ids = {}
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        ids[a.GetIdx()] = obj(0x8004)
        prop(0x0200, struct.pack("<ii", int(-p.y * 14.4 * 65536), int(p.x * 14.4 * 65536)))
        if a.GetAtomicNum() != 6:
            prop(0x0402, struct.pack("<h", a.GetAtomicNum()))
        if a.GetFormalCharge():
            prop(0x0421, struct.pack("<b", a.GetFormalCharge()))
        obj(0x8006)                                  # a text label inside the node: skipped with its contents
        prop(0x0700, b"label")
        out.extend(b"\x00\x00")
        out.extend(b"\x00\x00")
    for b in mol.GetBonds():
        obj(0x8005)
        prop(0x0604, struct.pack("<I", ids[b.GetBeginAtomIdx()]))
        prop(0x0605, struct.pack("<I", ids[b.GetEndAtomIdx()]))
        order = {1: 1, 2: 2, 3: 4}[int(b.GetBondTypeAsDouble())]
        if order != 1:
            prop(0x0600, struct.pack("<H", order))
        out.extend(b"\x00\x00")
    out.extend(b"\x00\x00\x00\x00\x00\x00")         # fragment, page, document
    return bytes(out)


# ------------------------------------------------------------------------------------------------ formats
def test_mol2_with_several_molecules(tmp_path):
    p = tmp_path / "set.mol2"
    p.write_text(mol2_block(mol3d(ETHANOL), "ethanol") + "\n" + mol2_block(mol3d(TFA), "TFA")
                 + mol2_block(mol3d(PHENOL), "phenol"))
    seen = []
    items = inputs.read_file(p, log=seen.append)
    assert [i.id for i in items] == ["ethanol", "TFA", "phenol"]
    assert smiles_of(items) == [canon(ETHANOL), canon(TFA), canon(PHENOL)]
    assert items[0].source == "set.mol2#1" and items[0].mol.GetNumAtoms() == 9       # hydrogens kept
    assert items[0].meta[formats.READER] == "RDKit (MOL2)" and items[0].meta[formats.COORDS] == "3d"
    assert any("3 structure(s) read with RDKit (MOL2)" in m for m in seen)


def test_sdf_v3000(tmp_path):
    from rdkit import Chem

    sio = io.StringIO()
    w = Chem.SDWriter(sio)
    w.SetForceV3000(True)
    for smi, name in ((PFOA, "PFOA"), ("CC(=O)[O-].[Na+]", "sodium acetate")):
        m = mol3d(smi)
        m.SetProp("_Name", name)
        w.write(m)
    w.close()
    assert "V3000" in sio.getvalue()
    p = tmp_path / "v3.sdf"
    p.write_text(sio.getvalue())
    items = inputs.read_file(p)
    assert [i.id for i in items] == ["PFOA", "sodium_acetate"]
    assert smiles_of(items) == [canon(PFOA), canon("CC(=O)[O-].[Na+]")]
    single = tmp_path / "tfa.mol"
    single.write_text(Chem.MolToV3KMolBlock(mol3d(TFA)))
    assert smiles_of(inputs.read_file(single)) == [canon(TFA)]
    assert inputs.read_file(single)[0].meta[formats.COORDS] == "3d"


def test_marvin_mrv(tmp_path):
    from rdkit import Chem

    p = tmp_path / "tfa.mrv"
    p.write_text(Chem.MolToMrvBlock(mol3d(TFA)))
    items = inputs.read_file(p)
    assert len(items) == 1 and items[0].id == "tfa" and smiles_of(items) == [canon(TFA)]
    assert items[0].meta[formats.COORDS] == "3d"


def test_chemdraw_cdxml(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import rdChemDraw

    p = tmp_path / "pfoa.cdxml"
    p.write_text(rdChemDraw.MolToChemDrawBlock(Chem.MolFromSmiles(PFOA)))
    items = inputs.read_file(p)
    assert smiles_of(items) == [canon(PFOA)] and items[0].meta[formats.COORDS] == "2d"


@pytest.mark.parametrize("smi", [TFA, PHENOL, "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"])
def test_chemdraw_cdx_binary(tmp_path, smi):
    """RDKit's Python reader takes .cdx as CDXML; formats.cdx_to_cdxml turns the binary records into CDXML first."""
    p = tmp_path / "drawing.cdx"
    p.write_bytes(cdx_bytes(smi))
    items = inputs.read_file(p)
    assert sorted(smiles_of(items)) == sorted(canon(s) for s in [canon(smi)])
    assert items[0].meta[formats.READER].endswith("(ChemDraw CDX)")
    with pytest.raises(ValueError, match="VjCD0100"):
        formats.cdx_to_cdxml(b"<CDXML/>")


def test_cml_rdkit_written_and_array_form(tmp_path):
    from rdkit import Chem

    p = tmp_path / "pfba.cml"
    p.write_text(Chem.MolToCMLBlock(mol3d("OC(=O)C(F)(F)C(F)(F)C(F)(F)F")))
    items = inputs.read_file(p)
    assert smiles_of(items) == [canon("OC(=O)C(F)(F)C(F)(F)C(F)(F)F")] and items[0].meta[formats.COORDS] == "3d"
    # CML 1 array form, 2D coordinates, a formal charge, implicit hydrogens by hydrogenCount, two molecules
    p = tmp_path / "two.cml"
    p.write_text("""<?xml version="1.0"?>
<cml xmlns="http://www.xml-cml.org/schema">
 <molecule id="m1" title="methylammonium">
  <atomArray atomID="a1 a2" elementType="C N" formalCharge="0 1" hydrogenCount="3 3" x2="0 1.5" y2="0 0"/>
  <bondArray atomRef1="a1" atomRef2="a2" order="1"/>
 </molecule>
 <molecule id="m2" title="benzene">
  <atomArray>
   <atom id="a1" elementType="C" hydrogenCount="1"/><atom id="a2" elementType="C" hydrogenCount="1"/>
   <atom id="a3" elementType="C" hydrogenCount="1"/><atom id="a4" elementType="C" hydrogenCount="1"/>
   <atom id="a5" elementType="C" hydrogenCount="1"/><atom id="a6" elementType="C" hydrogenCount="1"/>
  </atomArray>
  <bondArray>
   <bond atomRefs2="a1 a2" order="A"/><bond atomRefs2="a2 a3" order="A"/><bond atomRefs2="a3 a4" order="A"/>
   <bond atomRefs2="a4 a5" order="A"/><bond atomRefs2="a5 a6" order="A"/><bond atomRefs2="a6 a1" order="A"/>
  </bondArray>
 </molecule>
</cml>""")
    items = inputs.read_file(p)
    assert [i.id for i in items] == ["methylammonium", "benzene"]
    assert smiles_of(items) == [canon("C[NH3+]"), canon("c1ccccc1")]
    assert items[0].meta[formats.COORDS] == "2d" and items[1].meta[formats.COORDS] == ""


def test_inchi_list_and_text(tmp_path):
    from rdkit import Chem

    pfoa_inchi = Chem.MolToInchi(Chem.MolFromSmiles(PFOA))
    p = tmp_path / "list.inchi"
    p.write_text(f"{pfoa_inchi} PFOA\n{Chem.MolToInchi(Chem.MolFromSmiles(TFA))}\n# a comment\nInChI=1S/garbage\n")
    seen = []
    items = inputs.read_file(p, log=seen.append)
    assert smiles_of(items) == [canon(PFOA), canon(TFA)]
    assert items[0].id == "PFOA" and items[1].id == Chem.MolToInchiKey(Chem.MolFromSmiles(TFA))
    assert len(seen) == 1 and "line 4" in seen[0] and "Not a valid InChI" in seen[0]
    # the Molecules box: an InChI is parsed locally, with or without an id (tab or spaces)
    assert smiles_of(inputs.parse_text(pfoa_inchi)) == [canon(PFOA)]
    assert inputs.parse_text(f"{pfoa_inchi}\tperfluorooctanoic acid")[0].name == "perfluorooctanoic acid"


def test_inchikey_goes_to_pubchem(monkeypatch):
    from quantabench.chem import pubchem

    asked = []

    def fake(name, timeout=15.0, client=None, namespace="name"):
        asked.append((name, namespace))
        return {"cid": 9554, "smiles": PFOA, "title": "Perfluorooctanoic acid"}

    monkeypatch.setattr(pubchem, "lookup", fake)
    key = "SNGREZUHAYWORS-UHFFFAOYSA-N"
    hit = inputs.parse_text(key)[0]
    assert asked == [(key, "inchikey")] and hit.smiles == PFOA and hit.id == key
    assert hit.name == "Perfluorooctanoic acid" and hit.meta["pubchem_cid"] == 9554
    assert inputs.parse_text(f"{key} PFOA")[0].id == "PFOA"
    assert inputs.looks_like_smiles(key) is False


def test_cif_molecular_crystal(tmp_path):
    """Molecules that cross the cell edges are made whole; one entry per unique molecule, with its count."""
    import ase.io
    import numpy as np
    from ase import Atoms

    def place(smi, shift):
        m = mol3d(smi, seed=3)
        pos = m.GetConformer().GetPositions() - m.GetConformer().GetPositions().mean(axis=0) + shift
        return [a.GetSymbol() for a in m.GetAtoms()], pos

    symbols, positions = [], []
    for smi, shift in (("CO", (0.2, 0.2, 0.2)), ("CO", (5.0, 5.0, 5.0)), ("O", (2.5, 7.5, 2.5))):
        s, p = place(smi, np.array(shift))
        symbols += s
        positions += list(p)
    cell = Atoms(symbols, positions=positions, cell=[10, 10, 10], pbc=True)
    cell.wrap()                                   # the first methanol is split across the cell edges
    path = tmp_path / "crystal.cif"
    ase.io.write(path, cell, format="cif")
    items = inputs.read_file(path)
    assert sorted(smiles_of(items)) == sorted([canon("CO"), canon("O")])
    by = {canon(i.smiles): i for i in items}
    assert by[canon("CO")].meta["n_in_cell"] == "2" and by[canon("O")].meta["n_in_cell"] == "1"
    assert all(i.meta[formats.COORDS] == "3d" for i in items)


def _xyz_lines(m) -> str:
    conf = m.GetConformer()
    return "\n".join(f"{a.GetSymbol():2s} {conf.GetAtomPosition(a.GetIdx()).x:12.6f} "
                     f"{conf.GetAtomPosition(a.GetIdx()).y:12.6f} {conf.GetAtomPosition(a.GetIdx()).z:12.6f}"
                     for a in m.GetAtoms())


@pytest.mark.parametrize("ext", [".gjf", ".com"])
def test_gaussian_input_uses_the_charge(tmp_path, ext):
    acetate = mol3d("CC(=O)[O-]")
    p = tmp_path / f"acetate{ext}"
    p.write_text(f"%chk=acetate.chk\n%mem=2GB\n# B3LYP/6-31+G(d)\n opt freq\n\nacetate anion\n\n-1 1\n"
                 f"{_xyz_lines(acetate)}\n\n")
    items = inputs.read_file(p)
    assert smiles_of(items) == [canon("CC(=O)[O-]")] and items[0].id == "acetate"
    assert items[0].meta[formats.READER] == "ASE + RDKit rdDetermineBonds (Gaussian input)"
    radical = mol3d("C[CH2]")
    p = tmp_path / f"ethyl{ext}"
    p.write_text(f"# UB3LYP/6-31G(d)\n\nethyl radical\n\n0 2\n{_xyz_lines(radical)}\n\n")
    items = inputs.read_file(p)
    assert items[0].multiplicity == 2 and smiles_of(items) == [canon("C[CH2]")]


def test_orca_input_and_a_complex(tmp_path):
    """A complex in one geometry: split by connectivity, each molecule with its own charge (total from the file)."""
    import numpy as np

    from quantabench.chem import formats as fm

    p = tmp_path / "tfa.inp"
    p.write_text(f"! B3LYP def2-SVP Opt\n%pal nprocs 4 end\n* xyz 0 1\n{_xyz_lines(mol3d(TFA))}\n*\n")
    assert smiles_of(inputs.read_file(p)) == [canon(TFA)]
    (tmp_path / "geom.xyz").write_text(f"{mol3d(ETHANOL).GetNumAtoms()}\n\n{_xyz_lines(mol3d(ETHANOL))}\n")
    p = tmp_path / "eth.inp"
    p.write_text("! XTB2\n* xyzfile 0 1 geom.xyz\n")
    assert smiles_of(inputs.read_file(p)) == [canon(ETHANOL)]
    acid, base = mol3d("CC(=O)[O-]"), mol3d("C[NH3+]")
    pos = np.vstack([acid.GetConformer().GetPositions(), base.GetConformer().GetPositions() + [8.0, 0, 0]])
    nums = [a.GetAtomicNum() for a in acid.GetAtoms()] + [a.GetAtomicNum() for a in base.GetAtoms()]
    mols, notes = fm.molecules_from_geometry(nums, pos, charge=0)
    from rdkit import Chem

    assert sorted(Chem.MolToSmiles(Chem.RemoveHs(m)) for m in mols) == sorted([canon("CC(=O)[O-]"), canon("C[NH3+]")])
    assert any("charges" in n for n in notes)
    with pytest.raises(ValueError, match="xyz"):
        (tmp_path / "bad.inp").write_text("! HF\n%coords\nend\n")
        inputs.read_file(tmp_path / "bad.inp")


@pytest.mark.skipif(os.name == "nt", reason="the fake obabel is a Python script with a #! line")
def test_open_babel_fallback(tmp_path, monkeypatch):
    """An extension QuantaBench does not read goes to the user's obabel (found as engines are: here through
    QUANTABENCH_OBABEL), converted to SDF in a temporary folder; the reader is reported."""
    from quantabench.chem import external

    fake = tmp_path / "obabel"
    fake.write_text(f"""#!{sys.executable}
import sys
from rdkit import Chem
args = sys.argv[1:]
if args == ["-V"]:
    print("Open Babel 3.1.1 -- Oct 21 2025")
    raise SystemExit(0)
src, out = args[0], args[args.index("-O") + 1]
w = Chem.SDWriter(out)
for line in open(src):
    smi, name = line.split()
    m = Chem.MolFromSmiles(smi)
    m.SetProp("_Name", name)
    w.write(m)
w.close()
print(f"2 molecules converted", file=sys.stderr)
""")
    fake.chmod(0o755)
    data = tmp_path / "set.xyzfoo"
    data.write_text(f"{TFA} TFA\n{ETHANOL} ethanol\n")
    monkeypatch.setenv("QUANTABENCH_OBABEL", "")
    monkeypatch.setattr(external.shutil, "which", lambda name: None)
    monkeypatch.setattr(external, "conda_env_prefixes", lambda: [])
    monkeypatch.setattr(external, "_env_bin_dirs", lambda prefix: [])
    external.clear_cache()
    with pytest.raises(ValueError, match="Open Babel"):
        inputs.read_file(data)
    monkeypatch.setenv("QUANTABENCH_OBABEL", str(fake))
    external.clear_cache()
    seen = []
    items = inputs.read_file(data, log=seen.append)
    external.clear_cache()
    assert [i.id for i in items] == ["TFA", "ethanol"] and smiles_of(items) == [canon(TFA), canon(ETHANOL)]
    assert items[0].source == "set.xyzfoo#1" and items[0].meta[formats.READER] == "Open Babel 3.1.1 (via SDF)"
    assert any("read with Open Babel 3.1.1" in m for m in seen)
