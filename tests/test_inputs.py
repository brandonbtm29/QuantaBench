import pytest
from conftest import PFOA

from quantabench.chem import inputs


def test_csv_with_dtxsid(tmp_path):
    p = tmp_path / "cohort.csv"
    p.write_text("DTXSID,PREFERRED_NAME,SMILES\nDTXSID8031865,PFOA," + PFOA + "\nDTXSID0000001,blank,\n")
    items = inputs.read_file(p)
    assert len(items) == 1 and items[0].id == "DTXSID8031865" and items[0].smiles == PFOA


def test_csv_missing_smiles(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        inputs.read_file(p)


def test_smiles_text_lines():
    items = inputs.parse_text(PFOA + " PFOA\nCCO ethanol\n# comment\n")
    assert [i.id for i in items] == ["PFOA", "ethanol"]


def test_xyz_and_sdf(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    Chem.MolToXYZFile(m, str(tmp_path / "eth.xyz"))
    w = Chem.SDWriter(str(tmp_path / "eth.sdf"))
    m.SetProp("DTXSID", "DTXSID9020584")
    w.write(m)
    w.close()
    assert inputs.read_file(tmp_path / "eth.xyz")[0].smiles == "CCO"
    assert inputs.read_file(tmp_path / "eth.sdf")[0].id == "DTXSID9020584"


def test_safe_id():
    assert inputs.safe_id("8:2 FTOH / test") == "8_2_FTOH_test"
