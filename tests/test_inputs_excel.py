import importlib.util

import pytest
from conftest import K_PFOS, PFOA

from quantabench.chem import inputs

openpyxl = pytest.importorskip("openpyxl")

PFOS_ACID = "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFBA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"
HEADER = ["INPUT", "FOUND_BY", "DTXSID", "PREFERRED_NAME", "CASRN", "SMILES", "QSAR_READY_SMILES",
          "MS_READY_SMILES"]
ROWS = [
    ["335-67-1", "CASRN", "DTXSID8031865", "Perfluorooctanoic acid", "335-67-1", PFOA, PFOA, PFOA],
    # salt: SMILES keeps the counter-ion (the pipeline strips it), QSAR-ready is the parent acid
    ["2795-39-3", "CASRN", "DTXSID8037706", "Potassium perfluorooctanesulfonate", "2795-39-3", K_PFOS,
     PFOS_ACID, PFOS_ACID],
    # SMILES blank but a QSAR-ready form exists: fall back to it
    ["375-22-4", "CASRN", "DTXSID4059916", "Perfluorobutanoic acid", "375-22-4", None, PFBA, PFBA],
    # no structure at all (CompTox uses N/A for mixtures and unregistered inputs)
    ["foo", "-", "N/A", "Unknown mixture", "N/A", "N/A", None, ""],
    [None] * 8,  # trailing empty row
]


def make_xlsx(path, sheet_title="Main Data", extra_sheet=True, header_offset=0):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_title
    for _ in range(header_offset):
        ws.append([None])
    ws.append(HEADER)
    for r in ROWS:
        ws.append(r)
    if extra_sheet:
        other = wb.create_sheet("Second")
        other.append(["DTXSID", "SMILES"])
        other.append(["DTXSID_ETH", "CCO"])
    wb.save(path)
    return path


def test_comptox_xlsx(tmp_path):
    p = make_xlsx(tmp_path / "pfas_cohort.xlsx")
    rep = inputs.read_table(p)
    ids = [i.id for i in rep.inputs]
    assert ids == ["DTXSID8031865", "DTXSID8037706", "DTXSID4059916"]
    pfoa, kpfos, pfba = rep.inputs
    assert pfoa.name == "Perfluorooctanoic acid" and pfoa.smiles == PFOA
    assert pfoa.meta["CASRN"] == "335-67-1" and pfoa.meta["smiles_column"] == "SMILES"
    assert pfoa.source == "pfas_cohort.xlsx:2"
    assert kpfos.smiles == K_PFOS and "." in kpfos.smiles  # salt read as given
    assert pfba.smiles == PFBA and pfba.meta["smiles_column"] == "QSAR_READY_SMILES"
    assert rep.n_rows == 4 and rep.skipped == [("row 5", "no structure")]
    assert rep.smiles_columns == {"SMILES": 2, "QSAR_READY_SMILES": 1}
    assert any("skipped 1 row" in w for w in rep.warnings)
    assert any("QSAR_READY_SMILES" in w for w in rep.warnings)


def test_read_file_logs_and_salt_is_desalted_by_pipeline(tmp_path):
    from quantabench.chem.standardize import prepare

    p = make_xlsx(tmp_path / "list.xlsx")
    logs = []
    items = inputs.read_file(p, log=logs.append)
    assert len(items) == 3 and any("skipped 1" in m for m in logs)
    prepared, _notes = prepare(items[1].to_mol(), 7.0, True, "rules")
    assert "K" not in {a.GetSymbol() for a in prepared.GetAtoms()}


def test_qsar_preference_and_sheet_choice(tmp_path):
    p = make_xlsx(tmp_path / "list.xlsx", header_offset=2)  # header need not be on row 1
    items = inputs.read_file(p, smiles_preference="qsar")
    assert [i.meta["smiles_column"] for i in items] == ["QSAR_READY_SMILES"] * 3
    assert items[1].smiles == PFOS_ACID and items[0].source == "list.xlsx:4"
    second = inputs.read_file(p, sheet="Second")
    assert [i.id for i in second] == ["DTXSID_ETH"]
    assert [i.id for i in inputs.read_file(p, sheet=1)] == ["DTXSID_ETH"]
    with pytest.raises(ValueError, match="Main Data"):
        inputs.read_file(p, sheet="Nope")
    explicit = inputs.read_file(p, smiles_col="ms_ready_smiles", id_col="CASRN")
    assert [i.id for i in explicit] == ["335-67-1", "2795-39-3", "375-22-4"]


def test_csv_uses_same_logic(tmp_path):
    p = tmp_path / "cohort.csv"
    p.write_text("DTXSID,PREFERRED_NAME,SMILES,QSAR_READY_SMILES\n"
                 f"DTXSID8031865,PFOA,{PFOA},{PFOA}\n"
                 f"DTXSID4059916,PFBA,,{PFBA}\n"
                 "DTXSID0000001,blank,N/A,\n"
                 f"DTXSID8031865,PFOA,{PFOA},{PFOA}\n", encoding="utf-8")  # exact duplicate row
    rep = inputs.read_table(p)
    assert [(i.id, i.name, i.meta["smiles_column"]) for i in rep.inputs] == [
        ("DTXSID8031865", "PFOA", "SMILES"), ("DTXSID4059916", "PFBA", "QSAR_READY_SMILES")]
    assert rep.skipped == [("DTXSID0000001", "no structure")]
    assert any("duplicate" in w for w in rep.warnings)


def test_unparsable_smiles_falls_back(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(f"DTXSID,SMILES,QSAR_READY_SMILES\nX1,C1CC,{PFBA}\nX2,C1CC,\n", encoding="utf-8")
    items = inputs.read_file(p)
    assert items[0].smiles == PFBA and items[0].meta["smiles_column"] == "QSAR_READY_SMILES"
    assert items[1].smiles == "C1CC"  # kept, so the run reports the parse failure instead of dropping it


def test_duplicate_ids_with_different_structures(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("ID,SMILES\nA,CCO\nA,CCN\n", encoding="utf-8")
    assert [i.id for i in inputs.read_file(p)] == ["A", "A_2"]


@pytest.mark.skipif(importlib.util.find_spec("xlrd") is not None, reason="xlrd installed: .xls is readable")
def test_legacy_xls_gives_clear_error(tmp_path):
    p = tmp_path / "old.xls"
    p.write_bytes(b"\xd0\xcf\x11\xe0not really")
    with pytest.raises(ValueError, match="Save As"):
        inputs.read_file(p)
