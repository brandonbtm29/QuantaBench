"""Spartan document reader (chem.io_spartan): synthetic Input/PropertyArchive text and a fake container.

The integration test at the end reads real documents only when QB_SPARTAN_SAMPLES names a folder of them.
"""

from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

import numpy as np
import pytest

from quantabench.chem import io_spartan

# CF3-CH3 (1,1,1-trifluoroethane), made up for the tests; continuation keyword lines as Spartan writes them
INPUT = """C MS_CONFP2_BW HF SMETHOD=MECH/MMFF WINDOWKJ=85\r
 CONFSTEP1:WINDOWKJ=15 CONFSTEP1:CONFSKEPT=30\r
TEST-1\r
0 1\r
 6   0.000000000   0.000000000   0.000000000\r
 6   1.520000000   0.000000000   0.000000000\r
 9  -0.450000000   1.270000000   0.000000000\r
 9  -0.450000000  -0.635000000   1.100000000\r
 9  -0.450000000  -0.635000000  -1.100000000\r
 1   1.900000000  -1.020000000   0.000000000\r
 1   1.900000000   0.510000000   0.880000000\r
 1   1.900000000   0.510000000  -0.880000000\r
ENDCART\r
ATOMLABELS\r
"C1"\r
"C2"\r
"F1"\r
"F2"\r
"F3"\r
"H1"\r
"H2"\r
"H3"\r
ENDATOMLABELS\r
HESSIAN\r
    1    1   16   16   16   13   13   13\r
    1    2    1\r
    1    3    1\r
    1    4    1\r
    1    5    1\r
    2    6    1\r
    2    7    1\r
    2    8    1\r
ENDHESS\r
BEGINPREFERENCES\r
 MM:CONF_SELECTION_RULE=12\r
ENDPREFERENCES\r
"""


def _xyz_lines(pos: np.ndarray) -> str:
    flat = np.asarray(pos, float).ravel()
    return "\n".join(" ".join(f"{v:.10f}" for v in flat[i:i + 6]) for i in range(0, len(flat), 6))


def _props(conf: np.ndarray, rel: list[float], *, indices=None, degen=None, extra: str = "") -> str:
    n_conf, m = conf.shape[0], conf.shape[1]
    idx = indices if indices is not None else list(range(1, m + 1))
    text = (f"\r\n\r\nPROP STRING CALCULATION_PRODNAME = \"SPARTAN'24\"\r\n"
            f"PROP VALUE CONFORMATIONAL_ENTROPY =    5.1234500000\r\n"
            f"PROP VALUE CONF_NUM_SAVED =    {n_conf:.10f}\r\n"
            f"PROP VALUE CONF_ENERGY_OFFSET =  -50.2500000000\r\n"
            f"PROP STRING CONFMULTI_METHODS 2 BEGIN\r\n \"MMFF94\"\r\n \"DLFF03\"\r\nEND\r\n"
            f"PROP VALUE CONF_ATOM_INDICES {len(idx)} BEGIN\r\n" + " ".join(f"{float(i):.10f}" for i in idx) + "\r\nEND\r\n"
            f"PROP VALUE CONF_COORDS {n_conf} {3 * m} BEGIN\r\n{_xyz_lines(conf)}\r\nEND\r\n"
            f"PROP VALUE CONF_ENERGY_LIST {n_conf} BEGIN\r\n" + " ".join(f"{e:.10f}" for e in rel) + "\r\nEND\r\n"
            f"PROP VALUE:BYATOM MM_CHARGE 8 BEGIN\r\n 0.1 0.2 -0.1 -0.1 -0.1 0.0 0.0 0.0\r\nEND\r\n"
            f"PROP VALUE:BYATOM:STATIC ORIGINAL_COORDS 8 3 BEGIN\r\n{_xyz_lines(np.zeros((8, 3)))}\r\nEND\r\n")
    if degen is not None:
        text += f"PROP VALUE CONF_DEGENCOUNT {n_conf} BEGIN\r\n" + " ".join(f"{g:.1f}" for g in degen) + "\r\nEND\r\n"
    return text + extra


def _input_positions() -> np.ndarray:
    return io_spartan.parse_input(INPUT).positions_A


def _rotated(pos: np.ndarray, angle: float, shift=(1.0, 2.0, 3.0)) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return pos @ r.T + np.asarray(shift)


def _doc(molecules: list[tuple[str, str]]) -> io_spartan.DictContainer:
    streams = {"Document": b"\x00binary", "Sheet/SheetData": b"\x00"}
    for k, (inp, props) in enumerate(molecules, 1):
        streams[f"Molecules/M{k:04d}/Input.gz"] = gzip.compress(inp.encode("latin-1"))
        if props is not None:
            streams[f"Molecules/M{k:04d}/PropertyArchive.gz"] = gzip.compress(props.encode("latin-1"))
        streams[f"Molecules/M{k:04d}/Geometry"] = b"\x10\x00\x00\xbd"
    return io_spartan.DictContainer(streams)


# ------------------------------------------------------------------------------------------ Input.gz
def test_parse_input_continued_keywords_label_charge_atoms_bonds():
    inp = io_spartan.parse_input(INPUT)
    assert inp.keywords == ("MS_CONFP2_BW HF SMETHOD=MECH/MMFF WINDOWKJ=85 CONFSTEP1:WINDOWKJ=15 "
                            "CONFSTEP1:CONFSKEPT=30")
    assert inp.label == "TEST-1"
    assert (inp.charge, inp.multiplicity) == (0, 1)
    assert inp.numbers == [6, 6, 9, 9, 9, 1, 1, 1]
    assert inp.positions_A.shape == (8, 3) and inp.positions_A[1, 0] == pytest.approx(1.52)
    assert inp.atom_labels[:3] == ["C1", "C2", "F1"]
    assert inp.bonds[0] == (0, 1, 1) and len(inp.bonds) == 7


def test_parse_input_single_keyword_line_and_fallback():
    simple = "SOPT DLFF03     \nMolecule1\n   0   1\n 9 0 0 0\n 6 1.3 0 0\nENDCART\n"
    inp = io_spartan.parse_input(simple)
    assert (inp.keywords, inp.label, inp.numbers) == ("SOPT DLFF03", "Molecule1", [9, 6])
    # keyword lines without the "C" continuation marker: found by the charge/multiplicity + atom lines
    odd = "OPT B3LYP 6-31G*\nCONTINUED KEYWORDS\nTEST-3\n-1 1\n 8 0 0 0\nENDCART\n"
    inp = io_spartan.parse_input(odd)
    assert (inp.label, inp.charge, inp.keywords) == ("TEST-3", -1, "OPT B3LYP 6-31G* CONTINUED KEYWORDS")
    with pytest.raises(ValueError):
        io_spartan.parse_input("SP\nlabel\nno charge line\n")


# ------------------------------------------------------------------------------------------ PropertyArchive.gz
def test_parse_properties_scalars_blocks_qualifiers():
    text = ("\r\n\r\nPROP STRING NAME_S = \"hello world\"\r\n"
            "PROP VALUE E_KCAL =  -12.3456789000\r\n"
            "PROP VALUE:STATIC TINY =   1.0000000E-24\r\n"
            "PROP VALUE M 2 3 BEGIN\r\n 1 2 3\r\n 4 5 6\r\nEND\r\n"
            "PROP VALUE:BYATOM:STATIC Q 3 BEGIN\r\n 0.1 -0.2 0.1\r\nEND\r\n"
            "PROP STRING KW 2 BEGIN\r\n \"SOPT\"\r\n \"DLFF03\"\r\nEND\r\n"
            "PROP VALUE BAD 2 2 BEGIN\r\n 1 2 3\r\nEND\r\n"
            "garbage line\r\n")
    p = io_spartan.parse_properties(text)
    assert p["NAME_S"].value == "hello world" and p["NAME_S"].is_scalar
    assert p["E_KCAL"].value == pytest.approx(-12.3456789)
    assert p["TINY"].qualifiers == ("STATIC",) and p["TINY"].value == pytest.approx(1e-24)
    assert p["M"].value.shape == (2, 3) and p["M"].value[1, 0] == 4
    assert p["Q"].qualifiers == ("BYATOM", "STATIC") and p["Q"].value.tolist() == [0.1, -0.2, 0.1]
    assert p["KW"].value == ["SOPT", "DLFF03"]
    assert p["BAD"].value.shape == (3,)            # value count does not match 2 x 2: kept flat


# ------------------------------------------------------------------------------------------ molecules
def test_molecule_conformers_lowest_energy_and_weights():
    ref = _input_positions()
    confs = np.stack([_rotated(ref, a) for a in (0.1, 0.5, 0.9)])
    props = _props(confs, [0.9, 1e-24, 2.5], degen=[1, 2, 1])
    doc = _doc([(INPUT, props)])
    (mol,) = list(io_spartan.iter_molecules(doc, "batch.spartan"))
    assert (mol.source_file, mol.molecule_index, mol.storage, mol.label) == ("batch.spartan", 1, "M0001", "TEST-1")
    assert mol.n_conformers == 3 and mol.best_conformer == 1
    assert np.allclose(mol.positions_A(), confs[1])
    assert np.allclose(mol.conformer_energies_kcal, np.array([0.9, 0.0, 2.5]) - 50.25)
    w = mol.boltzmann_weights()
    x = np.exp(-np.array([0.9, 0.0, 2.5]) / (0.0019872043 * 298.15)) * [1, 2, 1]
    assert np.allclose(w, x / x.sum())
    row = io_spartan.molecule_row(mol)
    assert row["n_conformers"] == 3 and row["spartan_conf_index"] == 2
    assert row["spartan_conf_energy_kcal"] == pytest.approx(-50.25)
    assert row["spartan_conf_method"] == "DLFF03" and row["spartan_conf_degeneracy"] == 2
    assert row["spartan_conformational_entropy"] == pytest.approx(5.12345)
    assert row["spartan_calculation_prodname"] == "SPARTAN'24"
    assert "spartan_conf_coords" not in row and "spartan_mm_charge" not in row   # arrays stay out of the table
    assert row["formula"] == "C2H3F3" and row["charge"] == 0 and row["multiplicity"] == 1
    assert mol.per_atom()["mm_charge"].shape == (8,)
    rows = io_spartan.conformer_rows(mol)
    assert [r["conformer"] for r in rows] == [1, 2, 3] and rows[2]["spartan_conf_rel_kcal"] == pytest.approx(2.5)


def test_partial_conformers_are_completed_rigidly():
    ref = _input_positions()
    full = _rotated(ref, 0.7)
    stored = [1, 2, 3, 4, 5, 6]            # H2 and H3 (atoms 7, 8) left out, as Spartan does for rotor groups
    props = _props(full[np.array(stored) - 1][None], [0.0], indices=stored)
    (mol,) = list(io_spartan.iter_molecules(_doc([(INPUT, props)])))
    assert mol.n_conformers == 1
    # a rigid copy of the input: the missing atoms land exactly where the rigid motion puts them
    assert np.allclose(mol.positions_A(), full, atol=1e-8)
    assert any("store 6 of 8 atoms" in w for w in mol.warnings)


def test_molecule_without_library_or_archive():
    doc = _doc([(INPUT, None)])
    (mol,) = list(io_spartan.iter_molecules(doc))
    assert mol.n_conformers == 1 and mol.best_conformer is None
    assert np.allclose(mol.positions_A(), _input_positions())
    row = io_spartan.molecule_row(mol)
    assert "spartan_conf_index" not in row and "PropertyArchive" in row["spartan_notes"]


def test_unreadable_molecule_is_reported_not_fatal():
    doc = _doc([(INPUT, None), ("garbage\n", None)])
    errors: list[str] = []
    mols = list(io_spartan.iter_molecules(doc, "x.spartan", errors))
    assert len(mols) == 1 and len(errors) == 1 and "M0002" in errors[0]
    with pytest.raises(ValueError):
        list(io_spartan.iter_molecules(doc, "x.spartan"))


def test_to_mol_uses_spartan_bonds():
    (mol,) = list(io_spartan.iter_molecules(_doc([(INPUT, None)])))
    m = io_spartan.to_mol(mol)
    from rdkit import Chem

    assert m.GetProp("qb_bonds") == "spartan"
    assert Chem.MolToSmiles(Chem.RemoveHs(m)) == "CC(F)(F)F"
    assert np.allclose(m.GetConformer().GetPositions(), _input_positions())


def test_to_result_ranks_conformers():
    ref = _input_positions()
    confs = np.stack([_rotated(ref, a) for a in (0.1, 0.5, 0.9)])
    (mol,) = list(io_spartan.iter_molecules(_doc([(INPUT, _props(confs, [0.9, 0.0, 2.5]))]), "b.spartan"))
    res = io_spartan.to_result(mol)
    assert res.status == "ok" and res.input.name == "TEST-1" and res.method == "spartan: DLFF03"
    assert [c.rank for c in res.conformers] == [0, 1, 2]
    assert [round(c.rel_kcal, 3) for c in res.conformers] == [0.0, 0.9, 2.5]
    assert sum(c.weight for c in res.conformers) == pytest.approx(1.0)
    assert np.allclose(res.mol.GetConformer(res.conformers[0].conf_id).GetPositions(), confs[1])


# ------------------------------------------------------------------------------------------ the table
def test_write_table_streams_documents_and_flags_duplicate_labels(tmp_path, monkeypatch):
    ref = _input_positions()
    confs = np.stack([_rotated(ref, a) for a in (0.1, 0.5)])
    other = INPUT.replace("TEST-1", "TEST-2")
    docs = {"a.spartan": _doc([(INPUT, _props(confs, [0.0, 1.0])), (other, None)]),
            "b.spartan": _doc([(INPUT, None)]),
            "broken.spartan": None}

    def fake_open(path):
        if docs[Path(path).name] is None:
            raise ValueError("not a Spartan document")
        return docs[Path(path).name]

    monkeypatch.setattr(io_spartan, "open_container", fake_open)
    table, sdf = tmp_path / "out" / "all.csv", tmp_path / "out" / "all.sdf"
    s = io_spartan.write_table(["a.spartan", "b.spartan", "broken.spartan"], table, sdf=sdf)
    assert (s.files, s.files_failed, s.molecules, s.rows, s.duplicate_labels) == (3, 1, 3, 3, 1)
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert list(rows[0])[:5] == list(io_spartan.KEY_COLUMNS)
    assert [(r["source_file"], r["molecule_index"], r["label"], r["duplicate_label"]) for r in rows] == [
        ("a.spartan", "1", "TEST-1", "1"), ("a.spartan", "2", "TEST-2", "0"), ("b.spartan", "1", "TEST-1", "1")]
    assert rows[0]["n_conformers"] == "2" and rows[2]["n_conformers"] == "1"
    assert "spartan_conformational_entropy" in rows[0] and rows[1]["spartan_conformational_entropy"] == ""
    assert not list(table.parent.glob("*.jsonl"))          # the temporary rows file is gone

    from rdkit import Chem

    mols = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False)]
    assert len(mols) == 3 and mols[0].GetProp("label") == "TEST-1" and mols[0].GetProp("source_file") == "a.spartan"
    assert np.allclose(mols[0].GetConformer().GetPositions(), confs[0], atol=1e-4)
    assert mols[0].GetAtomWithIdx(0).GetDoubleProp("spartan_mm_charge") == pytest.approx(0.1)

    s = io_spartan.write_table(["a.spartan"], tmp_path / "conf.csv", conformers="all")
    rows = list(csv.DictReader((tmp_path / "conf.csv").open(encoding="utf-8")))
    assert [(r["label"], r["conformer"]) for r in rows] == [("TEST-1", "1"), ("TEST-1", "2"), ("TEST-2", "")]
    assert all(r["duplicate_label"] == "0" for r in rows)  # conformer rows of one molecule are not duplicates


def test_cli_import_routes_spartan_documents(tmp_path, monkeypatch):
    pytest.importorskip("olefile")          # the CLI checks for it before reading (a .venv from before it)
    from quantabench import cli

    monkeypatch.setattr(io_spartan, "open_container", lambda path: _doc([(INPUT, None)]))
    table = tmp_path / "t.csv"
    rc = cli.main(["import", "x.spartan", "y.SPARTAN", "--table", str(table)])
    assert rc == 0
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert len(rows) == 2 and all(r["duplicate_label"] == "1" for r in rows)


def test_gui_read_files_routes_spartan(monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench.gui import import_action

    monkeypatch.setattr(io_spartan, "open_container", lambda path: _doc([(INPUT, None), ("garbage\n", None)]))
    results, errors = import_action.read_files(["batch.spartan"])
    assert [r.input.name for r in results] == ["TEST-1"] and len(errors) == 1


# ------------------------------------------------------------------------------------------ uncompressed streams,
# level of theory, job status (Spartan for macOS writes Input/PropertyArchive without .gz)
BODY = INPUT.split("TEST-1\r\n", 1)[1]          # "0 1", the atoms, ATOMLABELS, HESSIAN, ...


def _inp(label: str, keywords: str = "OPT PM6 FREQ", swap: int | None = None) -> str:
    """An Input text with the CF3-CH3 atoms; ``swap`` replaces the first F by that element (another structure)."""
    body = BODY if swap is None else BODY.replace(" 9  -0.450000000   1.270000000", f"{swap}  -0.450000000   1.270000000")
    return f" {keywords}\r\n{label}\r\n{body}"


def _commands(task: str, task_name: str, method: str, method_class: str, basis: str = "") -> str:
    return (f"# version : SPARTAN'24 1.3.0  1.3.0 Win/64b\r\n# rootKeys: '{task}' '{method}' '' '' ''\r\n"
            f"# Task    :   {task} '{task_name}'\r\n# Method  :   {method} '{method_class}'\r\n"
            f"# Basis   : {basis}\r\n# Method  :  {method} (2)\r\nRunInitializeJob\r\n  {task} {method}\r\nEndStep\r\n")


def _calculations(keywords: str) -> bytes:
    return b"\xe0\x01\x00\xbd\x08\x00\x00\x00\xb0\x00\x00\xbd%\x00\x00\x00" + keywords.encode() + b"\x00\x00\x00\x01"


JOBLOG_OK = (b"\xe0\x05\x00\xbd\x08\x00\x00\x00\x00\x03\x00\xbd\x8e\x00\x00\x00\x00\x00\x00\xaeNormal exit\x00"
             b"Successful completion\x00/Applications/Spartan/sp_semiemp\x00\x00\x00")
JOBLOG_FAILED = JOBLOG_OK + (b"\x03\x00\xbd\x93\x00\x00\x00\x02\x00\x00\xa2Abnormal exit\x00Geometry failed to "
                             b"converge\x00/Applications/Spartan/sp_semiemp\x00\x00")
JOBLOG_WINDOWS = b"\x00\x03\x00\xbd\x93\x00\x00\x00\x00\x00\x00\xaeNormal exit.\x00No Message Returned  \x00C:/WFQM.exe\x00"
PM6 = ("PROP VALUE E_HOMO =     -0.3800000000 \r\nPROP VALUE E_LUMO =    -0.0200000000 \r\n"
       "PROP VALUE DIPOLE_MAG =       2.5000000 \r\nPROP VALUE ZEROPOINT =   25.0 \r\nPROP VALUE H_TOT =   27.5 \r\n"
       "PROP VALUE G_THERMO =   20.5 \r\nPROP VALUE S_TOT =   0.0235 \r\nPROP VALUE CPKAREA =   120.5 \r\n"
       "PROP VALUE E_KCAL =   {e} \r\nPROP STRING:STATIC OPTAT = \"PM6\"\r\n")


def _pm6(e: float = -200.0) -> str:
    return PM6.replace("{e}", f"{e:.4f}")


def _mac_doc(molecules: list[dict], gz: bool = False) -> io_spartan.DictContainer:
    """Each molecule: {"Input": text, "PropertyArchive": text, "Commands": text, "Calculations": bytes, "JobLog":
    bytes}; text streams uncompressed unless ``gz`` (then Input.gz / PropertyArchive.gz)."""
    streams = {"Document": b"\x00", "Sheet/SheetData": b"\xa1\x00\x00\xbd\x14\x00\x00\x00Label\x00\x00\x00\xc8\x00\x00\x00"}
    for k, parts in enumerate(molecules, 1):
        for name, data in parts.items():
            if data is None:
                continue
            raw = data.encode("latin-1") if isinstance(data, str) else data
            if gz and name in ("Input", "PropertyArchive"):
                streams[f"Molecules/M{k:04d}/{name}.gz"] = gzip.compress(raw)
            else:
                streams[f"Molecules/M{k:04d}/{name}"] = raw
        streams[f"Molecules/M{k:04d}/Geometry"] = b"\x10\x00\x00\xbd"
    return io_spartan.DictContainer(streams)


def _pm6_mol(label: str, e: float = -200.0, *, swap=None, joblog=JOBLOG_OK) -> dict:
    return {"Input": _inp(label, swap=swap), "PropertyArchive": _pm6(e) if joblog is not JOBLOG_FAILED else
            "PROP VALUE:STATIC OPTENERGY =   -0.31 \r\n",
            "Commands": _commands("OPT", "Equilibrium Geometry", "PM6", "Semi-Empirical"),
            "Calculations": _calculations("OPT PM6 FREQ DOQSAR"), "JobLog": joblog}


def test_uncompressed_streams_level_status_and_qsar_columns():
    doc = _mac_doc([_pm6_mol("M0001"), _pm6_mol("M0002", joblog=JOBLOG_FAILED)])
    m1, m2 = io_spartan.iter_molecules(doc, "b_SemiEmp_PM6.spartan")
    assert m1.label == "M0001" and m1.n_atoms == 8 and m1.properties["CPKAREA"].value == pytest.approx(120.5)
    lev = m1.level
    assert (lev.task, lev.task_name, lev.method, lev.method_class, lev.basis, lev.solvent) == (
        "OPT", "Equilibrium Geometry", "PM6", "Semi-Empirical", "", "")
    assert lev.version == "SPARTAN'24 1.3.0 Win/64b" and lev.keywords == "OPT PM6 FREQ DOQSAR"
    assert lev.kind == "semiempirical" and lev.is_quantum and lev.describe() == "PM6 OPT"
    assert m1.status == "completed" and m2.status == "failed: Geometry failed to converge (sp_semiemp)"
    row = io_spartan.molecule_row(m1)
    hartree_ev = 27.211386245988
    assert row["homo_ev"] == pytest.approx(-0.38 * hartree_ev) and row["lumo_ev"] == pytest.approx(-0.02 * hartree_ev)
    assert row["gap_ev"] == pytest.approx(0.36 * hartree_ev)
    assert row["dipole_debye"] == 2.5 and row["zpe_kcal"] == 25.0 and row["h_corr_kcal"] == 27.5
    assert row["g_corr_kcal"] == 20.5 and row["entropy_cal_molK"] == pytest.approx(23.5)
    assert "spartan_e_homo" not in row and row["spartan_cpkarea"] == pytest.approx(120.5)    # CPK: not sasa_A2
    assert (row["spartan_method"], row["spartan_task"], row["spartan_status"]) == ("PM6", "OPT", "completed")
    assert "starting geometry" in row["spartan_notes"]
    assert io_spartan.molecule_row(m2)["spartan_status"].startswith("failed")


def test_level_of_dft_solvent_and_conformer_search():
    calc = _calculations("SOPT B3LYP 6-311+G** DOQSAR SOLVENT=WASSER")
    doc = _mac_doc([{"Input": _inp("M0001", "SOPT B3LYP 6-311+G** SOLVENT=WASSER"), "PropertyArchive": _pm6(),
                     "Commands": _commands("SOPT", "Single Point Energy", "B3LYP", "Density-Functional", "6-311+G**"),
                     "Calculations": calc, "JobLog": JOBLOG_WINDOWS}], gz=True)
    (m,) = io_spartan.iter_molecules(doc)
    assert (m.level.method, m.level.basis, m.level.solvent, m.level.kind) == ("B3LYP", "6-311+G**", "WASSER", "dft")
    assert m.level.describe() == "B3LYP/6-311+G** SOPT SOLVENT=WASSER" and m.status == "completed"
    # a conformer search: the final stage's method, not the dialog's HF; its MMFF dipole is not dipole_debye
    ref = _input_positions()
    props = _props(np.stack([ref]), [0.0], extra="PROP VALUE DIPOLE_MAG =   1.5 \r\nPROP VALUE E_HOMO =  -0.3 \r\n")
    doc = _mac_doc([{"Input": INPUT, "PropertyArchive": props,
                     "Commands": _commands("MS_CONFP2_BW", "Conformer Distribution (multi-step)", "HF", "Hartree-Fock")}])
    (m,) = io_spartan.iter_molecules(doc)
    assert m.level.is_conformer_search and m.level.method == "DLFF03" and m.level.kind == "conf"
    row = io_spartan.molecule_row(m)
    assert "dipole_debye" not in row and row["spartan_dipole_mag"] == 1.5 and "homo_ev" not in row
    # no Commands stream: the level from the Input keywords
    (m,) = io_spartan.iter_molecules(_doc([(INPUT.replace("C MS_CONFP2_BW HF", "C MS_CONFP2_BW MMFF"), None)]))
    assert (m.level.task, m.level.method, m.level.kind) == ("MS_CONFP2_BW", "MMFF", "conf")


def test_unreadable_molecules_are_counted_by_reason():
    doc = _mac_doc([_pm6_mol("M0001"), {"Commands": "# Task : OPT\r\n"}, {"Input": "garbage\r\n"}])
    errors: list[str] = []
    skipped: dict[str, int] = {}
    mols = list(io_spartan.iter_molecules(doc, "x.spartan", errors, skipped))
    assert len(mols) == 1 and skipped == {"no Input stream": 1, "no geometry in Input": 1}
    assert any("M0002: no Input stream" in e for e in errors) and all(r in io_spartan.SKIP_REASONS for r in skipped)
    doc.streams["Molecules/M0001/Input.gz"] = b"\x1f\x8b\x08\x00truncated"
    skipped = {}
    list(io_spartan.iter_molecules(doc, "x.spartan", [], skipped))
    assert skipped["Input stream not readable"] == 1


# ------------------------------------------------------------------------------------------ the structure join
def test_structure_key_ignores_conformation_bond_orders_and_atom_order():
    (a,) = io_spartan.iter_molecules(_doc([(INPUT, None)]))
    moved = INPUT
    for old, new in ((" 6   1.520000000", " 6   1.620000000"), (" 1   1.900000000  -1.020000000", " 1   2.100000000  -0.920000000")):
        moved = moved.replace(old, new)
    (b,) = io_spartan.iter_molecules(_doc([(moved.replace("    1    2    1\r", "    1    2    2\r"), None)]))
    (c,) = io_spartan.iter_molecules(_doc([(_inp("X", swap=17), None)]))
    assert io_spartan.structure_key(a) == io_spartan.structure_key(b) != io_spartan.structure_key(c)
    assert len(io_spartan.structure_key(a)) == 16
    # without Spartan's bonds: perceived from the geometry, same key
    no_bonds = INPUT.split("HESSIAN")[0]
    (d,) = io_spartan.iter_molecules(_doc([(no_bonds, None)]))
    assert d.input.bonds == [] and io_spartan.structure_key(d) == io_spartan.structure_key(a)
    assert io_spartan.has_own_label("TEST-1") and not io_spartan.has_own_label("M0018[1]")
    assert not io_spartan.has_own_label("M0003") and not io_spartan.has_own_label("")


def _join_docs():
    ref = _input_positions()
    base = _doc([(INPUT, _props(np.stack([ref, _rotated(ref, 0.4)]), [0.0, 1.0])),      # TEST-1: CF3-CH3
                 (_inp("TEST-2", "MS_CONFP2_BW MMFF", swap=17), None),                  # CF2Cl-CH3
                 (_inp("TEST-3", "MS_CONFP2_BW MMFF", swap=35), None),                  # CF2Br-CH3, twice:
                 (_inp("TEST-4", "MS_CONFP2_BW MMFF", swap=35), None)])                 # a salt and its acid
    pm6 = _mac_doc([_pm6_mol("M0001", -200.0), _pm6_mol("M0001[1]", -205.0),             # TEST-1, two conformers
                    _pm6_mol("M0002[1]", -300.0, swap=17), _pm6_mol("M0003", -400.0, swap=17, joblog=JOBLOG_FAILED),
                    _pm6_mol("M0004", swap=35),                                            # ambiguous
                    _pm6_mol("M0005", swap=53)])                                           # CF2I-CH3: unresolved
    dft = _mac_doc([{"Input": _inp("M0001", "SOPT B3LYP 6-31G*"), "PropertyArchive": _pm6(-9.0e5),
                     "Commands": _commands("SOPT", "Single Point Energy", "B3LYP", "Density-Functional", "6-31G*"),
                     "Calculations": _calculations("SOPT B3LYP 6-31G*"), "JobLog": JOBLOG_WINDOWS}], gz=True)
    return {"batch.spartan": base, "batch_SemiEmp_PM6.spartan": pm6, "batch_DFT.spartan": dft,
            "lonely_PM6.spartan": _mac_doc([_pm6_mol("M0001")])}


def test_label_join_by_structure_ambiguous_unresolved(tmp_path, monkeypatch):
    docs = _join_docs()
    monkeypatch.setattr(io_spartan, "open_container", lambda path: docs[Path(path).name])
    # the derived document first: the base document's labels are only known after pass 1
    names = ["batch_SemiEmp_PM6.spartan", "batch.spartan", "batch_DFT.spartan", "lonely_PM6.spartan"]
    s = io_spartan.write_table(names, tmp_path / "long.csv", wide=tmp_path / "wide.csv")
    assert s.molecules == s.rows == 12 and not s.skipped and not s.errors
    rows = list(csv.DictReader((tmp_path / "long.csv").open(encoding="utf-8")))
    got = [(r["source_file"], r["label"], r["label_resolved"], r["label_source"]) for r in rows]
    assert got[:6] == [("batch_SemiEmp_PM6.spartan", "M0001", "TEST-1", "structure"),
                       ("batch_SemiEmp_PM6.spartan", "M0001[1]", "TEST-1", "structure"),
                       ("batch_SemiEmp_PM6.spartan", "M0002[1]", "TEST-2", "structure"),
                       ("batch_SemiEmp_PM6.spartan", "M0003", "TEST-2", "structure"),
                       ("batch_SemiEmp_PM6.spartan", "M0004", "", "ambiguous"),
                       ("batch_SemiEmp_PM6.spartan", "M0005", "", "unresolved")]
    assert [r["label_source"] for r in rows[6:10]] == ["input"] * 4 and rows[10]["label_resolved"] == "TEST-1"
    assert rows[11]["label_source"] == "unresolved" and "no base document" in rows[11]["label_note"]
    assert "TEST-3, TEST-4" in rows[4]["label_note"] and "not in batch" in rows[5]["label_note"]
    assert s.label_sources == {"structure": 5, "ambiguous": 1, "unresolved": 2, "input": 4}
    assert s.duplicate_labels == 0                       # Spartan's M0001 in two documents is not a label
    assert rows[0]["structure_key"] == rows[6]["structure_key"] and rows[0]["spartan_method"] == "PM6"
    # the wide table: one row per resolved label, each level's best row under its prefix
    assert s.prefixes == {"pm6": "PM6 OPT", "conf_dlff03": "DLFF03 MS_CONFP2_BW", "conf_mmff": "MMFF MS_CONFP2_BW",
                          "dft": "B3LYP/6-31G* SOPT"}            # two conformer-search levels: never one column
    wide = {r["label_resolved"]: r for r in csv.DictReader((tmp_path / "wide.csv").open(encoding="utf-8"))}
    assert set(wide) == {"TEST-1", "TEST-2", "TEST-3", "TEST-4"} and s.wide_rows == 4
    t1 = wide["TEST-1"]
    assert t1["pm6_n_rows"] == "2" and t1["pm6_storage"] == "M0002"          # E_KCAL -205 < -200
    assert float(t1["pm6_homo_ev"]) == pytest.approx(-0.38 * 27.211386245988) and t1["pm6_level"] == "PM6 OPT"
    assert t1["dft_level"] == "B3LYP/6-31G* SOPT" and float(t1["dft_homo_ev"]) < 0
    assert t1["conf_dlff03_n_conformers"] == "2" and t1["conf_mmff_n_rows"] == ""
    t2 = wide["TEST-2"]                                  # the completed row wins over the failed lower one
    assert (t2["pm6_n_rows"], t2["pm6_n_failed"], t2["pm6_storage"]) == ("2", "1", "M0003")
    assert t2["pm6_spartan_status"] == "completed" and t2["dft_n_rows"] == ""
    assert wide["TEST-3"]["pm6_n_rows"] == "" and wide["TEST-3"]["formula"] == "C2H3BrF2"   # ambiguous: left out
    assert list(wide["TEST-1"])[:3] == ["label_resolved", "structure_key", "formula"]


def test_level_prefixes_never_mix_levels():
    p = io_spartan.level_prefixes({"PM6 OPT": ("semiempirical", "PM6"), "DLFF03 MS_CONFP2_BW": ("conf", "DLFF03"),
                                   "B3LYP/6-311+G** SOPT SOLVENT=WASSER": ("dft", "B3LYP")})
    assert p == {"PM6 OPT": "pm6", "DLFF03 MS_CONFP2_BW": "conf", "B3LYP/6-311+G** SOPT SOLVENT=WASSER": "dft"}
    p = io_spartan.level_prefixes({"B3LYP/6-31G* SOPT": ("dft", "B3LYP"), "WB97X-D/6-31G* SOPT": ("dft", "WB97X-D"),
                                   "MMFF MS_CONFP2_BW": ("conf", "MMFF"), "DLFF03 MS_CONFP2_BW": ("conf", "DLFF03")})
    assert p == {"B3LYP/6-31G* SOPT": "dft_b3lyp_6_31gs", "WB97X-D/6-31G* SOPT": "dft_wb97x_d_6_31gs",
                 "MMFF MS_CONFP2_BW": "conf_mmff", "DLFF03 MS_CONFP2_BW": "conf_dlff03"}
    p = io_spartan.level_prefixes({"PM6 OPT": ("semiempirical", "PM6"), "PM6 SOPT": ("semiempirical", "PM6")})
    assert p == {"PM6 OPT": "pm6_opt", "PM6 SOPT": "pm6_sopt"}           # same method, different task


def test_cli_import_wide(tmp_path, monkeypatch):
    pytest.importorskip("olefile")
    from quantabench import cli

    docs = _join_docs()
    monkeypatch.setattr(io_spartan, "open_container", lambda path: docs[Path(path).name])
    rc = cli.main(["import", "batch.spartan", "batch_SemiEmp_PM6.spartan", "--table", str(tmp_path / "t.csv"),
                   "--wide", str(tmp_path / "w.csv")])
    assert rc == 0
    wide = list(csv.DictReader((tmp_path / "w.csv").open(encoding="utf-8")))
    assert {r["label_resolved"] for r in wide} == {"TEST-1", "TEST-2", "TEST-3", "TEST-4"}
    assert "pm6_homo_ev" in wide[0] and "conf_dlff03_n_conformers" in wide[0]


# ------------------------------------------------------------------------------------------ real documents
SAMPLES = os.environ.get("QB_SPARTAN_SAMPLES", "")


@pytest.mark.skipif(not SAMPLES or not Path(SAMPLES).is_dir(), reason="QB_SPARTAN_SAMPLES not set")
def test_real_documents(tmp_path):
    files = sorted(Path(SAMPLES).rglob("*.spartan"))
    assert files, f"no .spartan files under {SAMPLES}"
    s = io_spartan.write_table(files, tmp_path / "all.csv", sdf=tmp_path / "all.sdf", wide=tmp_path / "wide.csv")
    assert s.files_failed == 0 and not s.errors and not s.skipped, (s.skipped, s.errors[:5])
    rows = list(csv.DictReader((tmp_path / "all.csv").open(encoding="utf-8")))
    assert len(rows) == s.rows == s.molecules > 0
    for r in rows:
        assert r["label"] and int(r["n_atoms"]) > 0 and int(r["n_conformers"]) >= 1
        assert r["label_source"] in io_spartan.LABEL_SOURCES and r["spartan_method"], r["source_file"]
        assert (r["label_source"] == "input") == io_spartan.has_own_label(r["label"])
        if r["label_source"] in ("input", "structure"):
            assert r["label_resolved"]
        if r["spartan_method"] in ("PM6", "B3LYP") and r["spartan_status"] == "completed":
            assert float(r["homo_ev"]) < float(r["lumo_ev"])      # mapped, in eV
    stems = {f.stem for f in files}
    for r in rows:                                   # a document with its base among the inputs resolves fully
        base = max((b for b in stems if Path(r["source_file"]).stem.startswith(b + "_")), key=len, default=None)
        if base and r["label_source"] == "unresolved":
            pytest.fail(f"{r['source_file']} {r['storage']}: {r['label_note']}")
    wide = list(csv.DictReader((tmp_path / "wide.csv").open(encoding="utf-8")))
    assert len(wide) == s.wide_rows == len({r["label_resolved"] for r in rows if r["label_resolved"]})
    # labels of a batch agree with the SDF it was built from, when that SDF lies next to it
    for f in files:
        sdf = f.with_suffix(".sdf")
        if sdf.is_file():
            names = [line.strip() for line, prev in _sdf_titles(sdf)]
            got = [r["label"] for r in rows if r["source_file"] == str(f)]
            assert got == names


def _sdf_titles(path: Path):
    first = True
    prev = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if first:
            yield line, prev
            first = False
        if line.strip() == "$$$$":
            first = True
        prev = line
