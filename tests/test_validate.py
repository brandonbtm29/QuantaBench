"""Validation against Spartan (chem.validate): structure keys and joins, units, statistics, the sampler, the
comparison's outputs and the CLI, on synthetic tables. The integration test at the end reads a real wide table
only when QB_SPARTAN_WIDE names one."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import pytest
from rdkit import Chem

from quantabench.chem import validate as val


def _ik(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _can(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


WIDE_HEADER = ["label_resolved", "label_source", "cas_candidates", "structure_key", "inchikey", "canonical_smiles",
               "formula", "smiles", "charge", "structure_charge", "charge_check", "multiplicity", "n_atoms",
               "conf_level", "conf_source_file",
               "pm6_level", "pm6_source_file", "pm6_spartan_status", "pm6_dipole_debye", "pm6_homo_ev",
               "pm6_spartan_e_homo", "pm6_lumo_ev", "pm6_spartan_cpkarea",
               "dft_level", "dft_source_file", "dft_spartan_status", "dft_dipole_debye", "dft_homo_ev"]
PM6 = "PM6 OPT"
DFT = "B3LYP/6-311+G** SOPT SOLVENT=WASSER"


def _row(label, smiles, formula, charge=0, *, keys=True, source="input", cands="", check="ok", qs=None,
         cohort="C1A", pm6=None, dft=None, e_homo=None, homo=None, pm6_status="completed", archived_cohort=None):
    r = dict.fromkeys(WIDE_HEADER, "")
    r.update(label_resolved=label, label_source=source, cas_candidates=cands, formula=formula, smiles=smiles,
             charge=charge, multiplicity=1, n_atoms=10, conf_level="DLFF03 MS_CONFP2_BW",
             conf_source_file=f"C:\\data\\Archive\\{archived_cohort}_P1.spartan" if archived_cohort
             else f"C:\\data\\{cohort}_P1.spartan")
    if keys:
        r.update(inchikey=_ik(smiles), canonical_smiles=_can(smiles), structure_charge=qs if qs is not None else charge,
                 charge_check=check)
    if pm6 is not None:
        r.update(pm6_level=PM6, pm6_source_file=f"C:\\data\\{cohort}_P1_SemiEmp_PM6.spartan", pm6_spartan_status=pm6_status,
                 pm6_dipole_debye=pm6, pm6_homo_ev="" if homo is None else homo,
                 pm6_spartan_e_homo="" if e_homo is None else e_homo, pm6_lumo_ev=1.0, pm6_spartan_cpkarea=120.0)
    if dft is not None:
        r.update(dft_level=DFT, dft_source_file=f"C:\\data\\{cohort}_P1_DFT.spartan", dft_spartan_status="completed",
                 dft_dipole_debye=dft, dft_homo_ev=-8.0)
    return r


def _write(path: Path, rows: list[dict], header=None) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header or list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _spartan(tmp_path: Path) -> Path:
    rows = [
        _row("76-05-1", "O=C(O)C(F)(F)F", "C2HF3O2", pm6=2.0, homo=-12.0, dft=3.0),                # TFA
        _row("", "O=C([O-])C(F)(F)F", "C2F3O2-", -1, source="ambiguous", cands="14477-72-6;2966-50-9",
             cohort="C3A", dft=5.0),                                                                # its anion
        _row("X-CHIRAL", "C[C@H](F)Cl", "C2H4ClF", pm6=1.5, e_homo=-0.45),                         # hartree only
        _row("51-92-3", "C[N+](C)(C)C", "C4H12N+", -1, qs=1, check="mismatch", cohort="C3A", dft=9.9),
        _row("TFMS-ANION", "OS(O)(O)C(F)(F)F", "CF3O3S", -1, keys=False, cohort="C3A", dft=4.0),  # an older row
        _row("FAILED-1", "CCO", "C2H6O", pm6=1.7, homo=-11.0, pm6_status="failed: SCF failed to converge"),
    ]
    return _write(tmp_path / "wide.csv", rows, WIDE_HEADER)


# ------------------------------------------------------------------------------------------------- keys
def test_species_keys_skeleton_and_smiles_keys():
    acid, anion = Chem.MolFromSmiles("O=C(O)C(F)(F)F"), Chem.MolFromSmiles("[O-]C(=O)C(F)(F)F")
    ka, kb = val.species_keys(acid), val.species_keys(anion)
    assert ka["inchikey"] == _ik("OC(=O)C(F)(F)F") and kb["canonical_smiles"] == "O=C([O-])C(F)(F)F"
    assert ka["inchikey"][:14] == kb["inchikey"][:14] and val.skeleton(ka["inchikey"]) != val.skeleton(kb["inchikey"])
    assert val.skeleton(_ik("C[C@H](F)Cl")) == val.skeleton(_ik("CC(F)Cl")) and _ik("C[C@H](F)Cl") != _ik("CC(F)Cl")
    # explicit H and 3D coordinates do not change the keys
    mh = Chem.AddHs(anion)
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mh, randomSeed=1)
    assert val.species_keys(mh) == kb and val.species_keys(None) == {"inchikey": "", "canonical_smiles": ""}
    k = val.keys_from_smiles("[O-]C(=O)C(F)(F)F", charge=-1, formula="C2F3O2-")
    assert (k["inchikey"], k["structure_charge"], k["charge_check"], k["key_source"]) == (kb["inchikey"], -1, "ok", "smiles")
    assert val.keys_from_smiles("C[N+](C)(C)C", charge=-1)["charge_check"] == "mismatch"
    # connectivity-only SMILES of an older import: its hydrogens are not the formula's -> no key
    k = val.keys_from_smiles("OS(O)(O)C(F)(F)F", charge=-1, formula="CF3O3S")
    assert k["inchikey"] == "" and "hydrogens" in k["key_source"]
    assert val.keys_from_smiles("C[S+](C)[C-]1C=CC=C1", strict=True)["inchikey"] == ""
    assert val.keys_from_smiles("not a smiles")["inchikey"] == "" and val.skeleton("short") == ""
    assert val.formula_counts("C8HF15O2-") == {"C": 8, "H": 1, "F": 15, "O": 2} and val.bare_formula("CF3O3S-") == "CF3O3S"
    assert [val.charge_class(q) for q in (0, -2, 1, "")] == ["neutral", "anion", "cation", ""]
    assert val.cohort_of("C:\\x\\C1A_P1_SemiEmp_PM6.spartan") == "C1A"
    assert val.cohort_of("/x/EPA CompTox Essential 76 PFAS_Conf.1005-73-8.spartan") == "EPA CompTox Essential 76 PFAS"


# ------------------------------------------------------------------------------------------------- units
def test_unit_conversions():
    assert val.convert(-0.5, "hartree", "eV") == pytest.approx(-13.605693123)
    assert val.convert(1.0, "e*A", "D") == pytest.approx(4.80320471)
    assert val.convert(1.0, "au", "D") == pytest.approx(2.541746473)
    assert val.convert(10.0, "bohr3", "A3") == pytest.approx(1.48184711)
    assert val.convert(23.060548, "kcal/mol", "eV") == pytest.approx(1.0)
    assert val.convert(2.0, "D", "e*A") == pytest.approx(2.0 / 4.80320471)
    with pytest.raises(ValueError):
        val.convert(1.0, "eV", "D")


# ------------------------------------------------------------------------------------------------- statistics
def test_stats_against_hand_computed_values():
    st = val.stats([1, 2, 3, 4], [1.5, 2, 2.5, 5], ["a", "b", "c", "d"], top=3)
    # errors 0.5, 0, -0.5, 1
    assert st.n == 4 and st.mae == pytest.approx(0.5) and st.mse == pytest.approx(0.25)
    assert st.rmse == pytest.approx(math.sqrt(0.375))
    # Sxy = 5.5, Sxx = 5, Syy = 7.25
    assert st.pearson_r == pytest.approx(5.5 / math.sqrt(5 * 7.25)) and st.spearman_rho == pytest.approx(1.0)
    assert st.slope == pytest.approx(1.1) and st.intercept == pytest.approx(0.0, abs=1e-12)
    assert st.r2 == pytest.approx(5.5 ** 2 / (5 * 7.25))
    assert [o[0] for o in st.outliers] == ["d", "a", "c"] and st.outliers[0][3] == pytest.approx(1.0)
    assert val.stats([1, float("nan")], [2, 3]).n == 1
    assert val.stats([], []).n == 0 and math.isnan(val.stats([1, 2], [1, 3]).pearson_r)     # r needs N >= 3
    assert val.stats([1, 2, 3], [3, 1, 2]).spearman_rho == pytest.approx(-0.5)


# ------------------------------------------------------------------------------------------------- joins
def _qb_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    summ = [
        {"id": "76-05-1", "name": "TFA", "status": "ok", "smiles": "OC(=O)C(F)(F)F", "charge": 0, "solvent": "",
         "method": "tblite-gfn2", "dipole_debye": 2.2},
        {"id": "2966-50-9", "name": "TFA anion", "status": "ok", "smiles": "[O-]C(=O)C(F)(F)F", "charge": -1,
         "solvent": "water", "method": "tblite-gfn2", "dipole_debye": 6.0},
        {"id": "X-CHIRAL", "name": "", "status": "ok", "smiles": "CC(F)Cl", "charge": 0, "solvent": "",
         "method": "aimnet2", "dipole_debye": 1.0},
        {"id": "51-92-3", "name": "", "status": "ok", "smiles": "C[N+](C)(C)C", "charge": 1, "solvent": "water",
         "method": "tblite-gfn2", "dipole_debye": 0.0},
        {"id": "TFMS-ANION", "name": "", "status": "ok", "smiles": "[O-]S(=O)(=O)C(F)(F)F", "charge": -1,
         "solvent": "water", "method": "tblite-gfn2", "dipole_debye": 4.5},
        {"id": "NOPE", "name": "", "status": "ok", "smiles": "CCCC", "charge": 0, "solvent": "", "method": "mmff",
         "dipole_debye": 0.1},
        {"id": "BROKEN", "name": "", "status": "failed", "smiles": "", "charge": 0, "solvent": "", "method": "mmff"},
        {"id": "FAILED-1", "name": "", "status": "ok", "smiles": "CCO", "charge": 0, "solvent": "",
         "method": "tblite-gfn2", "dipole_debye": 1.6},
    ]
    _write(run / "summary.csv", summ)
    desc = [{"id": "76-05-1", "formula": "C2HF3O2", "tblite_gfn2_dipole_debye": 2.5, "tblite_gfn2_homo_ev": -13.0,
             "sasa_A2_best": 150.0, "dipole_debye_best": 2.2, "vib_zpe_kcal": 20.0},
            {"id": "TFMS-ANION", "formula": "CF3O3S-", "tblite_gfn2_dipole_debye": 4.6},
            {"id": "X-CHIRAL", "formula": "C2H4ClF", "dipole_debye_best": 1.0, "tblite_gfn2_homo_ev": -12.0}]
    _write(run / "descriptors.csv", desc)
    (run / "76-05-1").mkdir()
    (run / "76-05-1" / "76-05-1_summary.json").write_text(json.dumps({"conformers": [{"charges": [-0.5, 0.2, 0.7]}]}))
    return run


def test_joins_by_structure_ambiguous_missing_and_id_cross_check(tmp_path):
    table = val.read_spartan(_spartan(tmp_path))
    assert table.keys_computed == 1 and table.rows[4].inchikey == ""            # the older row: no key from its smiles
    assert {p for p, lev in table.levels.items() if lev.quantum} == {"pm6", "dft"}
    assert table.levels["dft"].phase == "water" and table.levels["pm6"].phase == "gas" and not table.levels["conf"].quantum
    assert table.rows[2].values["pm6"]["homo"] == pytest.approx(-0.45 * 27.211386245988)   # hartree -> eV
    recs = {r.id: r for r in val.read_quantabench([_qb_run(tmp_path)])}
    assert "BROKEN" not in recs and recs["76-05-1"].phases["gas"]["charge_min_e"] == -0.5
    assert recs["2966-50-9"].phases.keys() == {"water"} and recs["76-05-1"].key_source == "smiles"
    m = {x.rec.id: x for x in val.match(list(recs.values()), table)}
    assert (m["76-05-1"].join, m["76-05-1"].id_check, m["76-05-1"].cohort) == ("inchikey", "same", "C1A")
    # the anion: the ambiguous row of its structure; its id is one of the CAS candidates
    assert (m["2966-50-9"].join, m["2966-50-9"].id_check) == ("inchikey", "same")
    assert m["2966-50-9"].rows[0].label_source == "ambiguous" and m["2966-50-9"].charge_class == "anion"
    assert (m["X-CHIRAL"].join, m["X-CHIRAL"].id_check) == ("inchikey-nostereo", "same")    # no R/S in QuantaBench
    assert m["51-92-3"].join == "inchikey" and "another charge" in m["51-92-3"].note
    assert (m["TFMS-ANION"].join, m["TFMS-ANION"].id_check) == ("id+formula", "same")
    assert m["NOPE"].join == "" and m["NOPE"].rows == []
    # an id that is Spartan's label of another structure: the acid's CAS on the anion
    anion = val.QBRecord("76-05-1", "", _ik("[O-]C(=O)C(F)(F)F"), _can("[O-]C(=O)C(F)(F)F"), -1, "C2F3O2-", "smiles",
                         "tblite-gfn2", {"water": {}})
    (x,) = val.match([anion], table)
    assert x.join == "inchikey" and x.id_check == "differs" and "other protonation state" in x.note
    lone = val.QBRecord("76-05-1", "", _ik("CCCCCC"), "CCCCCC", 0, "C6H14", "smiles", "mmff", {"gas": {}})
    (y,) = val.match([lone], table)
    assert y.rows == [] and y.id_check == "differs" and "no structure match" in y.note


def test_validate_outputs_pairs_units_and_shift(tmp_path):
    table = val.read_spartan(_spartan(tmp_path))
    run = _qb_run(tmp_path)
    paired = _write(tmp_path / "paired.csv", [
        {"id": "76-05-1", "name": "TFA", "smiles": "OC(=O)C(F)(F)F", "charge": 0, "gas_tblite_gfn2_dipole_debye": 2.5,
         "water_tblite_gfn2_dipole_debye": 3.1, "delta_tblite_gfn2_dipole_debye": 0.6}])
    res = val.validate(val.read_quantabench([run, paired]), table, tmp_path / "out", top=3, plots=True)
    keys = {p.key: p for p in res.pairs}
    assert {"gas_dipole_tblite_gfn2_vs_pm6", "gas_dipole_refine_vs_pm6", "gas_homo_tblite_gfn2_vs_pm6",
            "gas_area_rdkit_vs_pm6", "water_dipole_tblite_gfn2_vs_dft", "water_dipole_refine_vs_dft",
            "shift_dipole_tblite_gfn2_vs_dft-minus-pm6"} <= set(keys)
    assert not any("q_min" in k for k in keys)            # QuantaBench has charges, this Spartan table has none
    assert keys["gas_area_rdkit_vs_pm6"].comparison == "correlation"
    assert keys["shift_dipole_tblite_gfn2_vs_dft-minus-pm6"].comparison == "shift (different levels)"
    summ = {(s["pair"], s["group_type"], s["group"]): s for s in res.summary}
    gas = summ[("gas_dipole_tblite_gfn2_vs_pm6", "all", "all")]
    assert gas["n"] == 1 and gas["mae"] == pytest.approx(0.5) and gas["mean_signed_error"] == pytest.approx(0.5)
    homo = summ[("gas_homo_tblite_gfn2_vs_pm6", "all", "all")]                  # TFA and X-CHIRAL (hartree input)
    assert homo["n"] == 2 and homo["mae"] == pytest.approx((1.0 + abs(-12.0 + 0.45 * 27.211386245988)) / 2)
    refine = summ[("gas_dipole_refine_vs_pm6", "all", "all")]                    # FAILED-1's PM6 job is left out
    assert refine["n"] == 2 and "FAILED-1" not in refine["top_outliers"]
    water = summ[("water_dipole_refine_vs_dft", "all", "all")]                   # anion + id-joined row; not 51-92-3
    assert water["n"] == 2 and water["mae"] == pytest.approx((1.0 + 0.5) / 2)
    assert ("water_dipole_refine_vs_dft", "charge_class", "anion") in summ
    assert ("gas_dipole_refine_vs_pm6", "cohort", "C1A") in summ
    shift = summ[("shift_dipole_tblite_gfn2_vs_dft-minus-pm6", "all", "all")]   # QB 3.1-2.5 vs Spartan 3.0-2.0
    assert shift["n"] == 1 and shift["mean_signed_error"] == pytest.approx(0.6 - 1.0)
    rows = {r["id"]: r for r in csv.DictReader((tmp_path / "out" / "validation.csv").open(encoding="utf-8"))}
    assert float(rows["76-05-1"]["gas_dipole_tblite_gfn2_vs_pm6__diff"]) == pytest.approx(0.5)
    assert "NOPE" not in rows and rows["2966-50-9"]["spartan_labels"] == "14477-72-6;2966-50-9"
    join = {r["id"]: r for r in csv.DictReader((tmp_path / "out" / "join.csv").open(encoding="utf-8"))}
    assert join["NOPE"]["join"] == "unmatched" and join["76-05-1"]["phases"] == "gas+water"
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Join coverage" in report and "| inchikey |" in report and "51-92-3" in report
    pytest.importorskip("matplotlib")
    assert (tmp_path / "out" / "parity_gas_homo_tblite_gfn2_vs_pm6.png").stat().st_size > 1000


def test_paired_delta_sign_warning(tmp_path):
    table = val.read_spartan(_spartan(tmp_path))
    paired = _write(tmp_path / "paired.csv", [
        {"id": "76-05-1", "smiles": "OC(=O)C(F)(F)F", "charge": 0, "gas_tblite_gfn2_dipole_debye": 2.5,
         "water_tblite_gfn2_dipole_debye": 3.1, "delta_tblite_gfn2_dipole_debye": -0.6}])
    res = val.validate(val.read_quantabench([paired]), table, tmp_path / "out", plots=False)
    assert any("gas - water" in w for w in res.warnings)


def test_join_where_the_charge_is_drawn_on_another_atom(tmp_path):
    """A sulfonimide anion drawn N- by Spartan and O- in QuantaBench's input: standard InChI keeps them apart
    (/q-1 vs /p-1), the charge-location key joins them; another anion of the same size does not."""
    n_minus = "O=S1(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)C1(F)F"
    o_minus = "O=S1(=O)N=S(=O)([O-])C(F)(F)C(F)(F)C1(F)F"
    assert _ik(n_minus) != _ik(o_minus) and _ik(n_minus)[:14] != _ik(o_minus)[:14]
    assert val.charge_parent_key(n_minus) == val.charge_parent_key(o_minus) != ""
    assert val.charge_parent_key("OC(=O)C(F)(F)F") == ""
    table = val.read_spartan(_write(tmp_path / "w.csv", [
        _row("189217-62-7", n_minus, "C3F6NO4S2-", -1, cohort="C3A", dft=12.0),
        _row("OTHER", "O=C([O-])C(F)(F)C(F)(F)C(F)(F)F", "C4F7O2-", -1, cohort="C3A", dft=5.0)], WIDE_HEADER))
    paired = _write(tmp_path / "p.csv", [{"id": "189217-62-7", "smiles": o_minus, "charge": -1,
                                          "water_tblite_gfn2_dipole_debye": 11.0}])
    (m,) = val.match(val.read_quantabench([paired]), table)
    assert m.join == "charge-location" and [r.label for r in m.rows] == ["189217-62-7"] and m.id_check == "same"


def test_ion_dipoles_from_an_older_import_are_flagged(tmp_path):
    """A wide table without *_spartan_dipole_origin_debye has ion dipoles about Spartan's frame origin."""
    paired = _write(tmp_path / "anion.csv", [
        {"id": "14477-72-6", "smiles": "O=C([O-])C(F)(F)F", "charge": -1, "water_tblite_gfn2_dipole_debye": 4.2}])
    recs = val.read_quantabench([paired])
    old = val.validate(recs, val.read_spartan(_spartan(tmp_path)), tmp_path / "old", plots=False)
    assert any("frame origin" in w for w in old.warnings)
    header = WIDE_HEADER + ["dft_spartan_dipole_origin_debye"]
    rows = list(csv.DictReader((tmp_path / "wide.csv").open(encoding="utf-8")))
    new = val.validate(recs, val.read_spartan(_write(tmp_path / "wide2.csv", rows, header)), tmp_path / "new",
                       plots=False)
    assert not any("frame origin" in w for w in new.warnings)
    assert "centre of mass" in (tmp_path / "new" / "report.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------- the sampler
def _sampler_table(tmp_path: Path) -> val.SpartanTable:
    rows = []
    chain = "C(F)(F)"
    for k in range(40):                                   # C1A neutral, PM6 only; sizes from 3 to 42 heavy atoms
        smi = "OC(=O)" + chain * (k // 2 + 1) + ("C" * (k % 2)) + "F"
        rows.append(_row(f"N-{k}", smi, "", pm6=1.0 + k / 10, homo=-12.0, cohort="C1A"))
    for k in range(6):                                    # C3A anions, DFT only
        rows.append(_row(f"A-{k}", "[O-]C(=O)" + chain * (k + 1) + "F", "", -1, cohort="C3A", dft=5.0))
    for k in range(10):                                   # C4A neutral: 3 with both, the rest PM6 only
        rows.append(_row(f"B-{k}", "C" * (k + 2) + "O", "", pm6=1.0, homo=-11.0, cohort="C4A",
                         dft=2.0 if k < 3 else None, archived_cohort="C4C" if k == 0 else None))
    rows.append(_row("B-dup", "C" * 2 + "O", "", pm6=1.0, cohort="C4A"))                 # B-0's structure again
    rows.append(_row("SALT", "[Na+].[O-]C(=O)C(F)(F)F", "", 0, cohort="C1A", pm6=1.0))   # two fragments
    rows.append(_row("TIN", "C[Sn](C)(C)C", "", 0, cohort="C1A", pm6=1.0))               # Sn: not AIMNet2's
    rows.append(_row("BADQ", "C[N+](C)(C)C", "", -1, qs=1, check="mismatch", cohort="C3A", dft=1.0))
    rows.append(_row("NONE", "CCCl", "", cohort="C1A"))                                   # no PM6, no DFT
    for r in rows:                                        # formulas for the element and size filters
        m = Chem.AddHs(Chem.MolFromSmiles(r["smiles"]))
        from rdkit.Chem import rdMolDescriptors

        r["formula"] = rdMolDescriptors.CalcMolFormula(m)
    return val.read_spartan(_write(tmp_path / "s.csv", rows, WIDE_HEADER))


def test_sampler_is_reproducible_and_fills_quotas(tmp_path):
    table = _sampler_table(tmp_path)
    a = val.sample(table, n=20, seed=7)
    b = val.sample(table, n=20, seed=7)
    c = val.sample(table, n=20, seed=8)
    ids = [r.label for r in a.rows]
    assert ids == [r.label for r in b.rows] and len(ids) == 20 and len(set(ids)) == 20
    assert ids != [r.label for r in c.rows]
    q = a.design["quotas"]
    # three strata, water-filled: 6 anions all taken, the rest split evenly (7 each) between the other two
    assert q == {"C1A|neutral": {"available": 36, "quota": 7}, "C3A|anion": {"available": 6, "quota": 6},
                 "C4A|neutral": {"available": 10, "quota": 7}}
    assert sum(v["quota"] for v in q.values()) == a.design["n_selected"] == 20
    assert a.excluded == {"more than one fragment": 1, "elements": 1, "Spartan's charge differs from its structure": 1,
                          "no Spartan values (any)": 1, "same structure as another row": 1,
                          "size": 4}                       # the four longest chains: more than 60 heavy atoms
    # DFT first: the three C4A rows with both values are in, B-0 under C4A although an archive also has it
    assert {"B-0", "B-1", "B-2"} <= set(ids) and a.design["with_both"] == 3 and a.design["with_dft_water"] == 9
    assert next(r for r in a.rows if r.label == "B-0").cohort == "C4A"
    # spread over size bins within a stratum
    bins = {val.size_bin(r.n_heavy, val.DEFAULT_SIZE_BINS) for r in a.rows if r.cohort == "C1A"}
    assert len(bins) >= 3
    assert val._water_fill(30, {("A",): 50, ("B",): 5, ("C",): 30}) == {("A",): 13, ("B",): 5, ("C",): 12}
    assert val._water_fill(5, {("A",): 1, ("B",): 1}) == {("A",): 1, ("B",): 1}


def test_sampler_filters(tmp_path):
    table = _sampler_table(tmp_path)
    s = val.sample(table, n=100, require="both")
    assert sorted(r.label for r in s.rows) == ["B-0", "B-1", "B-2"]
    s = val.sample(table, n=100, require="dft", charges=["anion"])
    assert {r.label for r in s.rows} == {f"A-{k}" for k in range(6)}
    s = val.sample(table, n=100, elements=None, allow_fragments=True, allow_charge_mismatch=True, max_heavy=8)
    labels = {r.label for r in s.rows}
    assert {"TIN", "SALT", "BADQ"} <= labels and all(r.n_heavy <= 8 for r in s.rows)
    s = val.sample(table, n=100, strata=["cohort"], cohorts={"N-1": "MINE"})
    assert any(r.label == "N-1" for r in s.rows) and "MINE" in s.design["quotas"]
    rows = val.sample_rows(s, table, cohorts={"N-1": "MINE"})
    assert {"id", "smiles", "name", "cohort", "charge_class", "inchikey", "has_pm6", "has_dft_water"} <= set(rows[0])
    assert next(r for r in rows if r["id"] == "N-1")["cohort"] == "MINE"
    with pytest.raises(ValueError):
        val.sample(table, require="everything")


# ------------------------------------------------------------------------------------------------- CLI
def test_cli_sample_and_validate(tmp_path, capsys):
    from quantabench import cli

    table = _sampler_table(tmp_path)
    out = tmp_path / "pick.csv"
    assert cli.main(["sample", table.path, "-o", str(out), "-n", "12", "--seed", "3"]) == 0
    picked = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(picked) == 12 and json.loads(out.with_suffix(".design.json").read_text())["seed"] == 3
    assert "with DFT (water)" in capsys.readouterr().out
    # QuantaBench input: build --file reads these columns
    from quantabench.chem import inputs

    mols = inputs.read_csv(out)
    assert len(mols) == 12 and mols[0].smiles == picked[0]["smiles"] and mols[0].id == picked[0]["id"]
    run = _qb_run(tmp_path)
    assert cli.main(["validate", str(run), "--spartan", str(_spartan(tmp_path)), "--out", str(tmp_path / "v"),
                     "--no-plots"]) == 0
    text = capsys.readouterr().out
    assert "joined to Spartan rows" in text and "gas_dipole_tblite_gfn2_vs_pm6" in text
    assert (tmp_path / "v" / "summary.csv").is_file() and not list((tmp_path / "v").glob("*.png"))


# ------------------------------------------------------------------------------------------------- a real table
WIDE = os.environ.get("QB_SPARTAN_WIDE", "")


@pytest.mark.skipif(not WIDE or not Path(WIDE).is_file(), reason="QB_SPARTAN_WIDE not set")
def test_real_wide_table(tmp_path):
    table = val.read_spartan(WIDE)
    assert len(table.rows) > 1000 and any(lev.quantum and lev.phase == "gas" for lev in table.levels.values())
    keyed = sum(1 for r in table.rows if r.inchikey)
    assert keyed / len(table.rows) > 0.9
    s = val.sample(table)
    assert len(s.rows) == min(300, s.design["eligible"]) and len({r.inchikey for r in s.rows}) == len(s.rows)
    assert [r.index for r in val.sample(table).rows] == [r.index for r in s.rows]            # reproducible
    assert all(r.charge_check != "mismatch" for r in s.rows)
    assert all(r.has("gas", table.levels) or r.has("water", table.levels) for r in s.rows)
    # QuantaBench records made from the sample's SMILES with Spartan's own values join back by InChIKey with
    # their CAS id and compare with zero error
    recs = []
    for r in s.rows[:50]:
        vals = {}
        for p, v in r.values.items():
            lev = table.levels[p]
            if lev.quantum and "dipole" in v and r.status.get(p) == "completed":
                vals.setdefault(lev.phase, {})["tblite_gfn2_dipole_debye"] = str(v["dipole"])
        recs.append(val.QBRecord(r.label or f"struct_{r.inchikey[:14]}", "", *_keys(r.canonical_smiles),
                                 r.charge, r.formula, "smiles", "tblite-gfn2", vals))
    res = val.validate(recs, table, tmp_path / "out", plots=False)
    assert all(m.join in ("inchikey", "smiles") for m in res.matches)
    assert all(m.id_check == "same" for m in res.matches if m.rows[0].label)
    for srow in res.summary:
        if srow["group_type"] == "all" and srow["phase"] != "shift" and srow["qb_source"] == "tblite_gfn2":
            assert srow["mae"] == pytest.approx(0.0, abs=1e-9), srow["pair"]


def _keys(smiles: str) -> tuple[str, str]:
    k = val.keys_from_smiles(smiles)
    return k["inchikey"], k["canonical_smiles"]
