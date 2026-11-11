"""``chem.pubchem_targets`` and ``quantabench-cli pubchem targets``: one value per molecule from the long-form
experimental records, moved to the target temperature, with tiers, spreads and what was left out."""

import csv

import pytest

from quantabench.chem import pubchem_targets as pt


def rec(id="m1", value=None, t=None, cls="pure_liquid", ref="Haynes, W.M. (ed.). CRC Handbook of Chemistry and "
        "Physics. 95th Edition.", raw=None, review="", heading="Surface Tension", unit="mN/m"):
    return {"id": id, "input_cas": "", "pubchem_inchikey": f"KEY-{id}", "cid": "1", "heading": heading,
            "raw": raw if raw is not None else f"{value} mN/m at {t} °C", "value_std": "" if value is None else str(value),
            "unit_std": unit, "temperature_c": "" if t is None else str(t), "condition_class": cls,
            "needs_review": review, "source_name": "Hazardous Substances Data Bank (HSDB)", "reference": ref}


def one(records, **kw):
    rows = pt.targets(records, pt.Settings(**kw))
    assert len(rows) == 1
    return rows[0]


def test_default_slope_moves_a_20C_value_to_25C():
    r = one([rec(value=24.0, t=20)])
    assert r["status"] == "ok"
    assert r["value"] == pytest.approx(23.5)          # -0.1 mN/m per K x 5 K
    assert r["value_raw"] == 24.0 and r["t_raw_c"] == 20.0 and r["correction"] == pytest.approx(-0.5)
    assert r["slope_source"] == "default" and "default_slope" in r["flags"] and "single_reference" in r["flags"]
    assert r["tier"] == "compilation"


def test_own_slope_from_records_spanning_the_target():
    recs = [rec(value=26.2, t=0, ref="Kirk-Othmer"), rec(value=23.7, t=20, ref="Kirk-Othmer"),
            rec(value=21.2, t=40, ref="Kirk-Othmer")]
    r = one(recs)
    assert r["slope"] == pytest.approx(-0.125)
    assert r["slope_source"].startswith("own fit (3 records, 0-40")
    assert r["value"] == pytest.approx(23.7 - 0.125 * 5)
    assert r["tier"] == "other" and r["excluded_outside_window"] == 2 and r["n_records_window"] == 1


def test_implausible_own_slope_falls_back_to_the_default():
    recs = [rec(value=30.0, t=20, ref="A"), rec(value=31.0, t=40, ref="A")]       # rises with T
    r = one(recs, window=(20, 40))
    assert r["slope"] == pytest.approx(-0.1) and r["slope_source"].startswith("default (own fit 0.05")


def test_best_tier_wins_and_the_spread_over_all_references_is_kept():
    recs = [rec(value=28.0, t=25, ref="U.S. Coast Guard, CHRIS - Hazardous Chemical Data"),
            rec(value=24.9, t=25), rec(value=25.3, t=25, ref="O'Neil, M.J. (ed.). The Merck Index")]
    r = one(recs)
    assert r["tier"] == "compilation" and r["value"] == pytest.approx(24.9) and r["n_references_tier"] == 1
    assert r["n_references"] == 3 and r["spread_all"] == pytest.approx(3.1) and "spread>2" in r["flags"]
    assert r["value_all_median"] == pytest.approx(25.3)


def test_one_value_per_reference_nearest_the_target():
    recs = [rec(value=24.13, t=25), rec(value=25.28, t=10), rec(value=22.13, t=50)]   # one CRC row, three columns
    r = one(recs, window=(10, 30))
    assert r["n_references"] == 1 and r["value"] == pytest.approx(24.13) and r["t_raw_c"] == 25.0


def test_review_estimated_untimed_and_other_class_records_are_left_out_and_counted():
    recs = [rec(value=20.0, t=20, review="yes"), rec(value=25.0, t=20, raw="est 25 dynes/cm= 0.025 N/m @ 20 °C"),
            rec(value=22.0), rec(value=60.0, t=20, cls="aqueous_solution")]
    r = one(recs)
    assert r["status"] != "ok" and r["value"] == ""
    assert (r["excluded_review"], r["excluded_estimated"], r["excluded_no_temperature"], r["excluded_class"]) == \
        (1, 1, 1, 1)
    r2 = one(recs, allow_no_temperature=True)
    assert r2["status"] == "ok" and r2["value"] == pytest.approx(22.0) and "temperature_assumed" in r2["flags"]
    r3 = one(recs, include_review=True, include_estimated=True)
    assert r3["n_references"] == 1 and r3["n_records_window"] == 2     # one reference, two records in the window


def test_unit_slips_are_repaired_or_left_out():
    recs = [rec(value=38000.0, t=25, raw="39.63 N/m at 10 °C; 38.00 N/m at 25 °C")]      # N/m written for mN/m
    r = one(recs)
    assert r["value"] == pytest.approx(38.0) and "unit_repaired" in r["flags"]
    r = one([rec(value=0.04355, t=20, raw="43.55 dynes/cm; 0.04355 Newtons/meter at 20 °C")])
    assert r["value"] == pytest.approx(43.05)
    r = one([rec(value=322.4, t=25, raw="322.4 dynes/cm @ 25 °C")])                       # no unit to blame
    assert r["status"] == "only implausible values" and r["excluded_implausible"] == 1


def test_every_molecule_with_a_record_gets_a_row_in_input_order():
    recs = [rec(id="b", value=30.0, t=25), rec(id="a", value=10.0, t=90), rec(id="c", value=1.0, t=25,
                                                                             heading="Density", unit="g/cm3")]
    rows = pt.targets(recs, pt.Settings())
    assert [r["id"] for r in rows] == ["b", "a"]
    assert rows[1]["status"] == "no record within 20-30 °C"
    dens = pt.targets(recs, pt.Settings(heading="Density"))
    assert [r["id"] for r in dens] == ["c"] and dens[0]["slope"] == pytest.approx(-0.001)


def test_tiers_from_reference_text():
    assert pt.reference_tier("Riddick, J.A., W.B. Bunger, Sakano T.K. Techniques of Chemistry") == ("compilation", 1)
    assert pt.reference_tier("Daubert, T.E., R.P. Danner. Physical and Thermodynamic Properties") == ("compilation", 1)
    assert pt.reference_tier("U.S. Coast Guard, Department of Transportation. CHRIS") == ("chris", 3)
    assert pt.reference_tier("Tomlin CDS, ed. The e-Pesticide Manual") == ("other", 2)
    assert pt.is_estimated("25 DYNES/CM= 0.025 NEWTONS/M AT 20 °C (ESTIMATED)") and not pt.is_estimated("24 mN/m")


def test_cli_writes_the_table(tmp_path, capsys):
    from quantabench import cli

    src = tmp_path / "records.csv"
    rows = [rec(id="ethanol", value=22.39, t=20), rec(id="ethanol", value=21.97, t=25),
            rec(id="water-ish", value=72.0, t=20, cls="aqueous_solution")]
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "st25.csv"
    assert cli.main(["pubchem", "targets", str(src), "--out", str(out), "--window", "20,25"]) == 0
    text = capsys.readouterr().out
    assert "1 of 2 molecule(s) with a target value" in text
    got = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [g["id"] for g in got] == ["ethanol", "water-ish"]
    assert float(got[0]["value"]) == pytest.approx(21.97) and got[0]["tier"] == "compilation"
    assert got[1]["status"] == "no record in the wanted condition class"
    assert cli.main(["pubchem", "targets", str(src), "--out", str(out), "--window", "20"]) == 2
