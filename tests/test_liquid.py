"""DESC-3, the feature builder: the dimensionless liquid-state groups (chem.liquid), the Analyze table's feature
presets (analysis.features) that write the same groups as formula columns, joined CSV files and the parity stats.

The GUI side (the Features menu, Join..., the 1:1 plot, persistence) is in tests/test_gui_table.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantabench.analysis import features as feat
from quantabench.analysis import formula as fm
from quantabench.analysis import sheet
from quantabench.chem import liquid as lq

# hexane: Bosque α 11.76 Å³ (0.32 + 1.51·6 + 0.17·14), V_m 131.6 cm³/mol (0.655 g/cm³), Bondi V_vdW 113.4 Å³
HEXANE = {"alpha": 11.76, "mu": 0.0, "V": 113.4, "A": 150.0, "Vm": 131.6}


def test_groups_of_hexane():
    g = lq.groups(HEXANE["alpha"], HEXANE["mu"], HEXANE["V"], HEXANE["A"], HEXANE["Vm"])
    assert list(g) == list(lq.COLUMNS)
    assert g["refractive_index_bosque"] == pytest.approx(1.375, abs=0.01)          # measured 1.375
    assert g["dipole_orientation_ratio"] == 0.0
    assert g["packing_fraction_vdw"] == pytest.approx(113.4 / (131.6 * 1e24 / lq.AVOGADRO))
    vm = 131.6 * 1e24 / lq.AVOGADRO * 1e-30                                        # m³ per molecule
    assert g["sigma_scale_kT_mN_m"] == pytest.approx(lq.BOLTZMANN * 298.15 / vm ** (2 / 3) * 1e3)
    assert 15 < g["sigma_lifshitz_mN_m"] < 25                                       # measured 17.9 mN/m
    # Israelachvili's Table 13.2 gives about 40 zJ for hexane across vacuum
    assert g["hamaker_lifshitz_zJ"] == pytest.approx(41.4, abs=2.0)


def test_sphericity_of_a_sphere_and_missing_inputs():
    r = 3.0
    g = lq.groups(10.0, 1.0, 4 / 3 * math.pi * r ** 3, 4 * math.pi * r ** 2, 100.0)
    assert g["sphericity_vdw"] == pytest.approx(1.0)
    assert lq.groups(None, None, None, None, None) == {}
    only_vm = lq.groups(None, None, None, None, 100.0)
    assert set(only_vm) == {"sigma_scale_kT_mN_m"}
    no_mu = lq.groups(10.0, float("nan"), 80.0, 100.0, 100.0)
    assert "lorentz_lorenz_bosque" in no_mu and "dipole_orientation_ratio" not in no_mu
    assert "hamaker_lifshitz_zJ" not in no_mu and "sigma_lifshitz_mN_m" not in no_mu
    # a dipole raises the static part of the Hamaker constant, a temperature lowers the orientation term
    polar = lq.groups(10.0, 2.0, 80.0, 100.0, 100.0)
    apolar = lq.groups(10.0, 0.0, 80.0, 100.0, 100.0)
    assert polar["hamaker_lifshitz_zJ"] > apolar["hamaker_lifshitz_zJ"]
    hot = lq.groups(10.0, 2.0, 80.0, 100.0, 100.0, temperature_k=350.0)
    assert hot["dipole_orientation_ratio"] < polar["dipole_orientation_ratio"]
    assert lq.orientation_A3_per_D2() == pytest.approx(8.10, abs=0.01)


def test_from_descriptors_prefers_the_boltzmann_dipole():
    d = {"polarizability_bosque_A3": 10.0, "dipole_debye_boltz": 1.0, "dipole_debye_best": 3.0,
         "vdw_volume_A3_best": 80.0, "vdw_area_A2_best": 100.0, "molar_volume_packing_cm3": 100.0}
    assert lq.from_descriptors(d) == lq.groups(10.0, 1.0, 80.0, 100.0, 100.0)
    d.pop("dipole_debye_boltz")
    assert lq.from_descriptors(d) == lq.groups(10.0, 3.0, 80.0, 100.0, 100.0)


def _evaluate(plan, env: dict, n: int) -> dict:
    for name, expr, _note in plan:
        env[name] = fm.parse(expr).evaluate(env, n=n)
    return env


def test_presets_compute_the_same_groups_as_chem_liquid():
    rng = np.random.default_rng(3)
    n = 40
    env = {"polarizability_bosque_A3": rng.uniform(3, 60, n), "dipole_debye_boltz": rng.uniform(0, 5, n),
           "vdw_volume_A3_best": rng.uniform(40, 500, n), "vdw_area_A2_best": rng.uniform(60, 600, n),
           "molar_volume_packing_cm3": rng.uniform(40, 600, n)}
    env["molar_volume_packing_cm3"][0] = 20.0                      # LL above the cap: both cap it the same way
    names = [p.name for p in feat.PRESETS if p.group == feat.GROUPS]
    plan = feat.plan(names, list(env))
    assert [p[0] for p in plan] == ["LL_bosque", "n_bosque", "ORI", "eta_vdw", "psi_vdw", "A_H_zJ", "sigma_H",
                                    "sigma_scale"]
    got = _evaluate(plan, dict(env), n)
    for i in range(n):
        want = lq.groups(*(env[k][i] for k in ("polarizability_bosque_A3", "dipole_debye_boltz", "vdw_volume_A3_best",
                                                "vdw_area_A2_best", "molar_volume_packing_cm3")))
        for p in feat.PRESETS:
            if p.descriptor:
                assert got[p.name][i] == pytest.approx(want[p.descriptor], rel=2e-6), (p.name, i)


def test_plan_reuses_descriptors_and_explains_what_is_missing():
    aliases = ["polarizability_bosque_A3", "vdw_volume_A3_best", "lorentz_lorenz_bosque", "molar_volume_packing_cm3"]
    plan = feat.plan(["n_bosque"], aliases)
    assert [p[0] for p in plan] == ["n_bosque"] and "lorentz_lorenz_bosque" in plan[0][1]
    assert feat.status("LL_bosque", aliases) == "already a column: lorentz_lorenz_bosque"
    assert feat.status("alpha_per_V", aliases) == ""
    with pytest.raises(feat.FeatureError, match="needs a dipole"):
        feat.plan(["ORI"], aliases)
    assert "van der Waals area" in feat.status("psi_vdw", aliases)
    assert feat.plan(["alpha_per_V"], aliases, formulas=["alpha_per_V"]) == []       # already made: reused
    with pytest.raises(feat.FeatureError, match="already the name of a column"):
        feat.plan(["alpha_per_V"], aliases + ["alpha_per_V"])                      # a column that is not the preset
    # a paired run's columns: the gas phase's
    paired = ["gas_polarizability_bosque_A3", "water_polarizability_bosque_A3", "gas_molar_volume_packing_cm3"]
    assert feat.plan(["LL_bosque"], paired)[0][1].endswith("*gas_polarizability_bosque_A3/gas_molar_volume_packing_cm3")


def test_reduced_target():
    aliases = ["st25", "sigma_scale_kT_mN_m", "molar_volume_packing_cm3"]
    assert feat.reduced_target("st25", aliases) == [
        ("st25_star", "st25/sigma_scale_kT_mN_m", feat.reduced_target("st25", aliases)[0][2])]
    steps = feat.reduced_target("st25", ["st25", "molar_volume_packing_cm3"])          # an older run: the scale too
    assert [s[0] for s in steps] == ["sigma_scale", "st25_star"] and steps[1][1] == "st25/sigma_scale"
    with pytest.raises(feat.FeatureError, match="no column"):
        feat.reduced_target("nope", aliases)
    env = {"st25": np.array([17.9]), "molar_volume_packing_cm3": np.array([131.6])}
    got = _evaluate(steps, env, 1)
    assert got["st25_star"][0] == pytest.approx(17.9 / lq.groups(None, None, None, None, 131.6)["sigma_scale_kT_mN_m"])


def test_pipeline_writes_the_groups_and_the_index_tracks_measured_ones():
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=2, rmsd_threshold=0.1, xtb_properties=False,
                               protonate=False, write_files=False, threads=1)
    # measured n_D at 20-25 °C: hexane 1.375, perfluorohexane 1.252, benzene 1.501
    for smiles, n_d in (("CCCCCC", 1.375), ("FC(F)(F)" + "C(F)(F)" * 4 + "C(F)(F)F", 1.252), ("c1ccccc1", 1.501)):
        res = pipeline.run_molecule(from_smiles(smiles, id="x"), cfg)
        assert res.status == "ok"
        d = res.descriptors
        assert {"lorentz_lorenz_bosque", "refractive_index_bosque", "packing_fraction_vdw", "sphericity_vdw",
                "sigma_scale_kT_mN_m"} <= set(d)
        assert d["refractive_index_bosque"] == pytest.approx(n_d, abs=0.025), smiles
        assert 0.45 < d["packing_fraction_vdw"] < 0.65 and 0.5 < d["sphericity_vdw"] < 1.0
        assert "dipole_orientation_ratio" not in d               # MMFF without xTB properties: no dipole


def test_parity_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([1.5, 2.0, 2.5, 5.0, 7.0])
    s = fm.parity_stats(x, y)
    assert s["n"] == 4 and s["mae"] == pytest.approx(0.5) and s["bias"] == pytest.approx(0.25)
    assert s["rmse"] == pytest.approx(math.sqrt((0.25 + 0 + 0.25 + 1) / 4))
    assert s["r2"] == pytest.approx(1 - 1.5 / 5.0)
    assert fm.parity_stats(np.array([1.0]), np.array([1.0])) is None


def test_join_reads_a_csv_by_id(tmp_path):
    p = tmp_path / "targets.csv"
    p.write_text("﻿CAS;sigma;rho;note\n 64-17-5 ;22.0;0.789;ok\n110-54-3;17.9;;x\n;1;2;no id\n"
                 "110-54-3;17.89;0.655;again\n", encoding="utf-8")
    assert sheet.csv_columns(p) == ["CAS", "sigma", "rho", "note"]
    got = sheet.read_join(sheet.JoinSpec(str(p), "CAS", ["sigma", "rho"]))
    assert got.rows == 4 and got.duplicates == 1 and got.no_id == 1
    assert got.block.ids == ["64-17-5", "110-54-3"]
    np.testing.assert_allclose(got.block.columns["sigma"], [22.0, 17.89])
    np.testing.assert_allclose(got.block.columns["rho"], [0.789, 0.655])
    every = sheet.read_join(sheet.JoinSpec(str(p), "CAS"))
    assert every.block.names == ["sigma", "rho", "note"] and every.block.columns["note"].dtype == object
    with pytest.raises(ValueError, match="no column 'id'"):
        sheet.read_join(sheet.JoinSpec(str(p), "id"))
    with pytest.raises(ValueError, match="no column 'density'"):
        sheet.read_join(sheet.JoinSpec(str(p), "CAS", ["density"]))
    assert sheet.JoinSpec(str(p), "CAS").label == "targets"
    st = sheet.TableState(joins=[sheet.JoinSpec(str(p), "CAS", ["sigma"])])
    back = sheet.TableState.from_json(st.to_json())
    assert back.joins == st.joins
    assert sheet.TableState.from_json('{"version": 1, "joins": [{"path": 3}]}').joins == []
