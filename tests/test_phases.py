"""Phases (chem.phases, cli_phases): one input in the gas phase and in water, paired per molecule by id and
structure key, never by row order."""

from __future__ import annotations

import csv
import sys

import pytest
from rdkit import Chem

from quantabench.chem import phases
from quantabench.chem.descriptors import graph_descriptors

KCAL = phases.KCAL_PER_EV


def test_parse_phases():
    assert phases.parse_phases("gas,water") == ["", "water"]
    assert phases.parse_phases("gas, H2O ,octanol") == ["", "water", "octanol"]
    assert phases.parse_phases(["none", "water"]) == ["", "water"]
    for bad in ("gas,gas", "water,h2o", "", "gas,lava"):
        with pytest.raises(ValueError):
            phases.parse_phases(bad)
    assert phases.phase_dirs("/x", ["", "water"]) == {"gas": phases.Path("/x/gas"), "water": phases.Path("/x/water")}


def test_graph_columns_are_the_2d_descriptors():
    assert set(phases.GRAPH_COLUMNS) == set(graph_descriptors(Chem.AddHs(Chem.MolFromSmiles("CCO"))))


def _row(mol_id, key, energy, status="ok", **kw):
    row = {"id": mol_id, "name": mol_id.upper(), "status": status, "input_inchikey": key, "inchikey": key,
           "canonical_smiles": "C", "input_smiles": "C", "charge": -1, "multiplicity": 1, "method": "tblite-gfn2",
           "best_energy_eV": energy, "n_conformers": 3, "seconds": 1.5, "formula": "C2H3O2-", "mol_weight": 59.0,
           "sasa_A2_best": 100.0, "notes": "", "phase_notes": "gas"}
    row.update(kw)
    return row


def _tables():
    gas = phases.PhaseTable("gas", {r["id"]: r for r in (
        _row("a", "KEY-A", -100.0, vib_gibbs_kcal=-2000.0),
        _row("b", "KEY-B", -200.0),
        _row("c", "KEY-C", -300.0),
        _row("d", "KEY-D", -400.0, status="failed", error="boom"),
        _row("e", "KEY-E", -500.0),
    )})
    water = {"solvent": "water", "solvation_model": "alpb", "phase_notes": "refine: ALPB(water)"}
    wat = phases.PhaseTable("water", {r["id"]: r for r in (       # another order than the gas phase
        _row("c", "KEY-C", -300.1, **water, method="xtb-gfn2"),
        _row("b", "KEY-OTHER", -200.2, **water),
        _row("a", "KEY-A", -100.5, vib_gibbs_kcal=-2011.0, tblite_gfn2_gsolv_kcal=-11.0, sasa_A2_best=98.0, **water),
        _row("d", "KEY-D", -400.3, **water),
        _row("f", "KEY-F", -600.0, **water),
    )})
    return gas, wat


def test_pairing_joins_by_id_and_structure_and_says_why_a_pair_has_no_dg():
    rows, columns = phases.pair_rows(list(_tables()))
    by_id = {r["id"]: r for r in rows}
    assert [r["id"] for r in rows] == ["a", "b", "c", "d", "e", "f"]
    a = by_id["a"]
    assert a["pair_status"] == "ok" and a["pair_notes"] == ""
    assert a["dG_solv_kcal"] == pytest.approx(-0.5 * KCAL)                  # E_water(R_w) - E_gas(R_g)
    assert a["dG_solv_G_kcal"] == pytest.approx(-11.0)                      # G_water - G_gas
    assert a["dG_solv_vertical_kcal"] == pytest.approx(-11.0)               # the water run's E(ALPB) - E(gas)
    assert a["delta_sasa_A2_best"] == pytest.approx(-2.0) and a["delta_best_energy_eV"] == pytest.approx(-0.5)
    assert a["gas_status"] == a["water_status"] == "ok" and a["water_solvent"] == "water" and a["gas_solvent"] == ""
    assert "delta_n_conformers" not in a and "delta_seconds" not in a and "delta_charge" not in a
    assert a["charge"] == -1 and a["formula"] == "C2H3O2-" and "gas_formula" not in a     # identity: once
    b = by_id["b"]                                    # same id, different input structure: never paired
    assert "structure mismatch" in b["pair_status"] and "input structure differs" in b["pair_notes"]
    assert b["water_best_energy_eV"] == "" and "dG_solv_kcal" not in b
    c = by_id["c"]                                    # different methods: no solvation energy
    assert "dG_solv_kcal" not in c and "method differs" in c["pair_notes"]
    assert c["delta_best_energy_eV"] == pytest.approx(-0.1)
    assert by_id["d"]["pair_status"] == "gas failed" and "dG_solv_kcal" not in by_id["d"]
    assert by_id["e"]["pair_status"] == "water not computed" and by_id["e"]["water_status"] == ""
    assert by_id["f"]["pair_status"] == "gas not computed" and by_id["f"]["name"] == "F"
    # layout: identity, status, the solvation energies, then gas_/water_/delta_ per column
    assert columns[:3] == ["id", "name", "input_inchikey"]
    i = columns.index("pair_status")
    assert columns[i:i + 5] == ["pair_status", "pair_notes", "dG_solv_kcal", "dG_solv_G_kcal", "dG_solv_vertical_kcal"]
    j = columns.index("gas_sasa_A2_best")
    assert columns[j:j + 3] == ["gas_sasa_A2_best", "water_sasa_A2_best", "delta_sasa_A2_best"]
    assert "delta_notes" not in columns and "delta_method" not in columns and "gas_id" not in columns


def test_pairing_with_three_phases_names_each_difference():
    gas, wat = _tables()
    octa = phases.PhaseTable("octanol", {"a": _row("a", "KEY-A", -100.3, solvent="octanol")})
    rows, columns = phases.pair_rows([wat, gas, octa])        # the gas phase is the reference wherever it is
    a = next(r for r in rows if r["id"] == "a")
    assert a["dG_solv_water_kcal"] == pytest.approx(-0.5 * KCAL)
    assert a["dG_solv_octanol_kcal"] == pytest.approx(-0.3 * KCAL)
    assert "delta_octanol_best_energy_eV" in columns and "dG_solv_kcal" not in columns
    rows, _ = phases.pair_rows([gas, wat], reference="water")
    a = next(r for r in rows if r["id"] == "a")
    assert a["dG_solv_gas_kcal"] == pytest.approx(0.5 * KCAL) and "dG_solv_gas_vertical_kcal" not in a
    with pytest.raises(ValueError):
        phases.pair_rows([gas, wat], reference="octanol")


def test_a_water_run_that_fell_back_to_the_gas_phase_has_no_dg():
    gas, wat = _tables()
    wat.rows["a"]["solvent"] = ""
    rows, _ = phases.pair_rows([gas, wat])
    a = next(r for r in rows if r["id"] == "a")
    assert "dG_solv_kcal" not in a and "computed in the gas phase" in a["pair_notes"]


@pytest.fixture
def tb(monkeypatch):
    if "tblite" not in sys.modules:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return pytest.importorskip("tblite")


def _sdf(path):
    """A -2 phosphonate, a +1 ammonium, a sodium salt and a neutral molecule, with 3D coordinates."""
    from rdkit.Chem import AllChem

    w = Chem.SDWriter(str(path))
    for title, smi, cas in (("methylphosphonate", "C[P](=O)([O-])[O-]", "993-13-5"),
                            ("methylammonium", "C[NH3+]", ""),
                            ("sodium acetate", "[Na+].CC(=O)[O-]", "127-09-3"),
                            ("methanol", "CO", "67-56-1")):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=7)
        m.SetProp("_Name", title)
        if cas:
            m.SetProp("CAS_RN", cas)
        w.write(m)
    w.close()
    return path


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def test_build_phases_end_to_end(tb, tmp_path, monkeypatch, capsys):
    """build --phases gas,water: two normal run folders and paired.csv; the charges as drawn reach the engine in
    both phases (no pH model); pair rebuilds the table; --resume continues every phase."""
    from quantabench import cli
    from quantabench.chem import engines, jobs

    seen = []
    make = engines.make_calculator

    def spy(key, numbers, charge=0, multiplicity=1, device="cpu", **options):
        seen.append((len(numbers), charge, options.get("solvent", "")))
        return make(key, numbers, charge, multiplicity, device, **options)

    monkeypatch.setattr(engines, "make_calculator", spy)
    sdf = _sdf(tmp_path / "in.sdf")
    root = tmp_path / "out"
    args = ["build", "--file", str(sdf), "--id-col", "CAS_RN", "--method", "tblite-gfn2", "--no-protonate",
            "--max-confs", "2", "--refine-top", "1", "--threads", "1", "--lanes", "1", "--phases", "gas,water",
            "--run-dir", str(root)]
    assert cli.main(args) == 0, capsys.readouterr().out
    for label in ("gas", "water"):
        assert (root / label / jobs.DB_NAME).is_file() and (root / label / "summary.csv").is_file()
    st = jobs.open_run(root / "water")
    assert st.config["solvent"] == "water" and st.config["protonate"] is False and st.counts["ok"] == 4
    assert jobs.open_run(root / "gas").config["solvent"] == ""
    # the drawn charges, unchanged, in both phases: CH3PO3(2-) at -2, CH3NH3+ at +1, acetate (Na+ removed) at -1
    assert {(8, -2, ""), (8, -2, "water"), (8, 1, ""), (8, 1, "water"), (7, -1, ""), (6, 0, "water")} <= set(seen)
    gas_rows = _csv(root / "gas" / "summary.csv")
    assert gas_rows["993-13-5"]["charge"] == "-2" and gas_rows["methylammonium"]["charge"] == "1"
    assert gas_rows["127-09-3"]["removed_fragments"] == "[Na+]" and gas_rows["127-09-3"]["charge"] == "-1"
    assert "no CAS_RN property" in gas_rows["methylammonium"]["notes"]
    assert gas_rows["67-56-1"]["inchikey"] == "OKKJLVBELUTLKV-UHFFFAOYSA-N"
    desc = _csv(root / "water" / "descriptors.csv")
    assert list(desc["67-56-1"])[:4] == ["id", "inchikey", "canonical_smiles", "input_inchikey"]
    paired = _csv(root / "paired.csv")
    assert set(paired) == {"993-13-5", "methylammonium", "127-09-3", "67-56-1"}
    for row in paired.values():
        assert row["pair_status"] == "ok" and float(row["dG_solv_kcal"]) < 0, row["pair_notes"]
        assert float(row["dG_solv_vertical_kcal"]) < 0 and row["water_solvation_model"] == "alpb"
    assert float(paired["993-13-5"]["dG_solv_kcal"]) < float(paired["127-09-3"]["dG_solv_kcal"]) \
        < float(paired["67-56-1"]["dG_solv_kcal"])       # dianion < anion < neutral
    assert float(paired["67-56-1"]["delta_tblite_gfn2_dipole_debye"]) > 0
    # pair, from the folder build --phases wrote (its phase folders), into another file
    assert cli.main(["pair", str(root), "--out", str(tmp_path / "again.csv")]) == 0
    assert _csv(tmp_path / "again.csv").keys() == paired.keys()
    # resume: nothing left to do in either phase
    capsys.readouterr()
    assert cli.main(["build", "--phases", "gas,water", "--resume", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.count("4 of 4 molecule(s) already finished") == 2 and "4 molecule(s), 4 ok in every phase" in out


def test_resume_starts_a_phase_the_root_does_not_have_yet(tb, tmp_path, capsys):
    from quantabench import cli
    from quantabench.chem import jobs

    root = tmp_path / "out"
    base = ["--method", "tblite-gfn2", "--no-protonate", "--max-confs", "1", "--refine-top", "1", "--threads", "1",
            "--lanes", "1"]
    assert cli.main(["build", "CO", "--phases", "gas"]) == 2
    assert "at least two phases" in capsys.readouterr().err
    assert cli.main(["build", "CO", "--phases", "gas,water", "--solvent", "water"]) == 2
    # a gas-phase run made earlier with a plain build; --phases --resume adds the water phase from its inputs
    assert cli.main(["build", "CO methanol", "CCO ethanol", *base, "--solvent", "none", "--run-dir",
                     str(root / "gas")]) == 0
    capsys.readouterr()
    assert cli.main(["build", "--phases", "gas,water", "--resume", str(root)]) == 0
    assert "started" in capsys.readouterr().out
    st = jobs.open_run(root / "water")
    assert st.counts["ok"] == 2 and st.config["solvent"] == "water" and st.config["max_conformers"] == 1
    assert (root / "paired.csv").is_file()
