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


def _sdf(path, records):
    """records: (title, SMILES or None for a broken record, {property: value})."""
    from rdkit import Chem

    blocks = []
    for title, smi, props in records:
        if smi is None:   # a five-valent carbon: the connection table reads, sanitization fails
            m = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
            block = Chem.MolToMolBlock(m, kekulize=False)
        else:
            block = Chem.MolToMolBlock(Chem.MolFromSmiles(smi))
        block = title + block[block.index("\n"):]
        blocks.append(block + "".join(f">  <{k}>\n{v}\n\n" for k, v in props.items()) + "$$$$\n")
    path.write_text("".join(blocks))
    return path


def test_sdf_titles_are_ids_and_bad_records_are_reported(tmp_path):
    """CAS-number titles become ids unchanged; a record RDKit cannot read is reported with its record number,
    title and reason (it used to vanish without a word); a repeated title gets a suffix."""
    path = _sdf(tmp_path / "set.sdf", [("335-67-1", "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", {}),
                                       ("75-07-0", None, {}),
                                       ("2706-90-3", "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", {}),
                                       ("2706-90-3", "OC(=O)C(F)(F)C(F)(F)C(F)(F)F", {})])
    seen = []
    items = inputs.read_file(path, log=seen.append)
    assert [i.id for i in items] == ["335-67-1", "2706-90-3", "2706-90-3_2"]
    assert items[0].name == "335-67-1" and items[0].source == "set.sdf#1"
    assert any("record 2 (75-07-0)" in w and "valence" in w.lower() for w in seen), seen
    assert any("Duplicate id '2706-90-3'" in w for w in seen)


def test_sdf_id_property(tmp_path):
    """--id-col names an SD property for SD files too (it was ignored); records without it keep the title."""
    path = _sdf(tmp_path / "set.sdf", [("Trifluoroacetic acid", "OC(=O)C(F)(F)F", {"CAS_RN": "76-05-1"}),
                                       ("Acetic acid", "CC(=O)O", {})])
    seen = []
    items = inputs.read_file(path, id_col="cas_rn", log=seen.append)
    assert [i.id for i in items] == ["76-05-1", "Acetic_acid"]
    assert items[0].name == "Trifluoroacetic acid" and items[0].meta["CAS_RN"] == "76-05-1"
    assert any("1 record(s) have no cas_rn property" in w for w in seen)


def test_sdf_with_hydride_fragment_reads_quietly(tmp_path, capfd):
    path = _sdf(tmp_path / "h.sdf", [("hydride salt", "[H-].[Na+]", {})])
    capfd.readouterr()           # writing the test file warns too
    assert len(inputs.read_file(path)) == 1
    assert "not removing hydrogen" not in capfd.readouterr().err


def test_all_letter_smiles_are_not_sent_to_pubchem():
    from quantabench.chem.inputs import looks_like_smiles

    for smi in ("CCCO", "CCCC", "OCCO", "NCCN", "CCOCC", "ClCCl", "CCCCCCCCCCCCO", "CO", "c1ccccc1"):
        assert looks_like_smiles(smi), smi
    for name in ("benzene", "ethanol", "water", "PFOA", "PFOS", "GenX", "Bromine", "acetone"):
        assert not looks_like_smiles(name), name


def test_smiles_list_file_skips_bad_lines(tmp_path, monkeypatch):
    from quantabench.chem import inputs as inp

    def offline(name, timeout=15.0):
        raise ValueError(f"PubChem lookup failed for {name!r}: offline")

    monkeypatch.setattr(inp, "lookup_name", offline)
    p = tmp_path / "list.smi"
    p.write_text("OC(=O)C(F)(F)F TFA\nCCO ethanol\nC1CC(C\nnotarealname\nCCCO propanol\n")
    msgs = []
    items = inp.read_file(p, log=msgs.append)
    assert [m.id for m in items] == ["TFA", "ethanol", "propanol"]
    assert len(msgs) == 2 and "line 3" in msgs[0] and "line 4" in msgs[1]
    with pytest.raises(ValueError):            # a single entry (GUI box, CLI argument) still raises
        inp.parse_text("notarealname")
    bad = tmp_path / "bad.smi"
    bad.write_text("C1CC(C\n")
    with pytest.raises(ValueError, match="No molecule could be read"):
        inp.read_file(bad)
