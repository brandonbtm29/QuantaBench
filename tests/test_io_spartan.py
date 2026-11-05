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


# ------------------------------------------------------------------------------------------ real documents
SAMPLES = os.environ.get("QB_SPARTAN_SAMPLES", "")


@pytest.mark.skipif(not SAMPLES or not Path(SAMPLES).is_dir(), reason="QB_SPARTAN_SAMPLES not set")
def test_real_documents(tmp_path):
    files = sorted(Path(SAMPLES).rglob("*.spartan"))
    assert files, f"no .spartan files under {SAMPLES}"
    s = io_spartan.write_table(files, tmp_path / "all.csv", sdf=tmp_path / "all.sdf")
    assert s.files_failed == 0 and not s.errors, s.errors
    rows = list(csv.DictReader((tmp_path / "all.csv").open(encoding="utf-8")))
    assert len(rows) == s.rows == s.molecules > 0
    for r in rows:
        assert r["label"] and int(r["n_atoms"]) > 0 and int(r["n_conformers"]) >= 1
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
