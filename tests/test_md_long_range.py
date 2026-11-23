"""MD-LR: the long-range options of a liquid slab, the slab-safe pppm/disp rule and the recommendation
(``md.long_range``). Pure Python: no LAMMPS, no numpy."""

import math

import pytest

from quantabench.md import long_range as LR
from quantabench.md import recommend_long_range

PILOT_SLAB = (40.33, 40.33, 200.90)          # pilot A's slab box (Å)


def test_spacing_rule_reproduces_a_fine():
    """G·rc 2.77 and h·G 0.52 give A-fine's G (0.2305 1/Å in its slab log) and its 18 x 18 x 90 grid for A's box."""
    g = LR.rule_gewald(12.0)
    assert g == pytest.approx(0.2308, abs=1e-4) and abs(g - 0.2305425) / 0.2305425 < 0.002
    h = LR.rule_spacing(g)
    assert h == pytest.approx(2.2527, abs=1e-3)
    assert LR.disp_mesh(PILOT_SLAB, h) == (18, 18, 90)
    assert LR.disp_mesh((40.33, 40.33, 100.9), h) == (18, 18, 45)
    assert LR.disp_mesh((26.469314, 26.469314, 118.690352), h) == (12, 12, 54)   # 53 rounded up, as LAMMPS does
    assert LR.rule_gewald(10.0) == pytest.approx(0.277)
    # the real-space r⁻⁶ left at rc: pilot A's G·rc (0.1709 x 12) about 21 %, the rule's 1.8 %
    assert LR.real_space_left(0.1708603 * 12) == pytest.approx(0.21, abs=0.005)
    assert LR.real_space_left(2.77) == pytest.approx(0.0177, abs=5e-4)
    with pytest.raises(ValueError):
        LR.rule_gewald(5.0)
    with pytest.raises(ValueError):
        LR.disp_mesh(PILOT_SLAB, 0.0)


def test_settings_names_and_descriptions():
    e = LR.settings("ewald")
    assert e == {"method": "ewald", "dispersion": "ewald", "cutoff": 12.0, "kspace_accuracy": 1e-6,
                 "disp_grid": "spacing", "disp_gewald": pytest.approx(0.230833, rel=1e-5),
                 "disp_spacing": pytest.approx(2.2527, rel=1e-4)}
    c = LR.settings("cutoff+tail")
    assert c == {"method": "cutoff", "dispersion": "cutoff16", "cutoff": 16.0, "kspace_accuracy": 1e-5}
    a = LR.settings("full", disp_grid="accuracy")
    assert (a["disp_real_accuracy"], a["disp_kspace_accuracy"]) == (1e-5, 2e-5) and "disp_gewald" not in a
    assert LR.dispersion_name("cutoff", 14) == "cutoff14" and LR.dispersion_name("ewald", 10) == "ewald"
    assert LR.method_of("cutoff12") == "cutoff" and LR.method_of("ewald") == "ewald"
    assert "18 x 18 x 90" in LR.describe(e, PILOT_SLAB) and "pppm/disp" in LR.describe(e)
    assert "lj/cut/coul/long 16 + pppm, Coulomb 1e-05" in LR.describe(c)
    assert LR.config_kwargs(c) == {"dispersion": "cutoff16", "cutoff": 16.0, "kspace_accuracy": 1e-5}
    for bad in ("pme", ""):
        with pytest.raises(ValueError, match="long-range method"):
            LR.normalize_method(bad)
    with pytest.raises(ValueError, match="screening length"):
        LR.settings("cutoff", 40.0)
    opts = LR.options()
    assert opts["cutoff"]["label"] == "LJ cutoff (screening length)" and opts["cutoff"]["range"] == (8.0, 30.0)
    assert set(opts["method"]["choices"]) == {"ewald", "cutoff"} and opts["provisional"] is True


def test_recommendation_follows_the_pilots():
    gpu = recommend_long_range("slab", PILOT_SLAB, n_molecules=489, n_atoms=9780, machine=True)
    assert (gpu.method, gpu.cutoff, gpu.accelerator, gpu.dispersion) == ("cutoff", 16.0, "gpu", "cutoff16")
    assert gpu.ns_per_day == pytest.approx(21.0) and "11.71" in gpu.reason and "11.25" in gpu.reason
    assert gpu.summary().startswith("Recommended for this setup: 16 Å LJ cutoff + tail correction on the GPU because")
    assert gpu.provisional and gpu.summary().endswith("[provisional]")
    assert any("2.0 %" in n for n in gpu.notes) and not gpu.warnings
    assert gpu.config_kwargs() == {"dispersion": "cutoff16", "cutoff": 16.0, "kspace_accuracy": 1e-5}

    cpu = recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=LR.MachineInfo(False, "test"))
    assert (cpu.method, cpu.cutoff, cpu.accelerator) == ("ewald", 12.0, "cpu")
    assert cpu.settings == LR.settings("ewald") and cpu.ns_per_day == pytest.approx(6.6)
    assert "3.9" in cpu.reason and "6.6" in cpu.reason

    ref = recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=True, purpose="reference")
    assert (ref.method, ref.accelerator) == ("ewald", "cpu") and ref.settings["kspace_accuracy"] == 1e-6
    assert any("GPU stays idle" in n for n in ref.notes)

    narrow = recommend_long_range("interface", (30.0, 30.0, 180.0), machine=True)   # 16 Å needs > 33 Å
    assert narrow.method == "ewald" and "too narrow" in narrow.reason

    tiny = recommend_long_range("slab", (20.0, 20.0, 120.0), machine=False)
    assert tiny.cutoff == 9.0 and tiny.warnings and tiny.settings["disp_gewald"] == pytest.approx(2.77 / 9)

    bulk = recommend_long_range("bulk", (40.0, 40.0, 100.0), n_atoms=9780, machine=True)
    assert (bulk.method, bulk.cutoff, bulk.accelerator) == ("cutoff", 12.0, "gpu") and bulk.ns_per_day == 13.4
    assert not any("tail correction is added" in n for n in bulk.notes)
    assert recommend_long_range("bulk", machine=False).ns_per_day is None

    # the speed scales with the atoms (by molecules when only they are known)
    assert recommend_long_range("slab", PILOT_SLAB, n_atoms=2 * 9780, machine=True).ns_per_day == pytest.approx(10.5)
    half = recommend_long_range("slab", PILOT_SLAB, n_molecules=978, machine=False)
    assert half.ns_per_day == pytest.approx(3.3) and "978 molecules" in half.speed_basis
    d = gpu.to_dict()
    assert d["dispersion"] == "cutoff16" and d["summary"] == gpu.summary() and "A-fine" in d["provisional_note"]
    with pytest.raises(ValueError):
        recommend_long_range("droplet")
    with pytest.raises(ValueError):
        recommend_long_range(purpose="fast")


def test_choice_against_recommendation_and_warnings():
    rec = recommend_long_range("slab", PILOT_SLAB, n_atoms=9780, machine=True)
    assert LR.differences(LR.settings("cutoff", 16), rec) == []
    assert LR.differences(LR.settings("ewald"), rec) == ["method: ewald (recommended: cutoff)",
                                                          "LJ cutoff (screening length): 12 Å (recommended: 16 Å)"]
    assert LR.differences(LR.settings("cutoff", 16, kspace_accuracy=1e-6), rec) == [
        "Coulomb PPPM accuracy: 1e-06 (recommended: 1e-05)"]
    # a 12 Å cutoff in a slab: pilot C
    w = LR.check_settings(LR.settings("cutoff", 12), "slab", PILOT_SLAB)
    assert len(w) == 1 and "4.7 %" in w[0] and "7.96" in w[0]
    assert LR.check_settings(LR.settings("cutoff", 12), "bulk") == []
    assert LR.check_settings(LR.settings("cutoff", 16), "slab") == []
    assert any("residual low bias" in x for x in LR.check_settings(LR.settings("cutoff", 16), purpose="reference"))
    # the old pppm/disp settings: LAMMPS' estimate in a slab
    old = LR.settings("ewald", disp_grid="accuracy", disp_real_accuracy=1e-4, disp_kspace_accuracy=2e-3)
    w = LR.check_settings(old, "slab")
    assert len(w) == 1 and "2 x 2 x 8" in w[0] and "−0.30" in w[0]
    assert LR.check_settings(LR.settings("ewald", disp_grid="accuracy"), "slab") == []
    assert LR.check_settings(LR.settings("ewald"), "slab", PILOT_SLAB) == []
    coarse = LR.check_settings(LR.settings("ewald", disp_gewald=0.171, disp_spacing=6.0), "slab")
    assert any("21 %" in x for x in coarse) and any("coarser than the mesh check validated" in x for x in coarse)
    assert any("looser than any pilot" in x for x in LR.check_settings(LR.settings("cutoff", 16, kspace_accuracy=1e-4)))
    assert any("too small" in x for x in LR.check_settings(LR.settings("cutoff", 16), "slab", (30.0, 30.0, 200.0)))
    assert LR.notes_for(LR.settings("ewald"), "slab", True) and not LR.notes_for(LR.settings("ewald"), "slab", False)


def test_pilot_table_is_documented():
    """One table holds every number the recommendations quote; each row says where it comes from."""
    for key, row in LR.PILOT_TABLE.items():
        assert row.source and row.method in LR.METHODS and row.accelerator in ("cpu", "gpu"), key
        assert row.ns_per_day is None or row.ns_per_day > 0
    t = LR.PILOT_TABLE
    assert (t["B"].sigma_tail, t["B"].sigma_err, t["B"].plateau_vs_bulk) == (11.71, 0.30, -0.020)
    assert (t["C"].sigma_raw, t["C"].sigma_tail, t["C"].plateau_vs_bulk) == (7.96, 11.12, -0.047)
    assert t["A"].sigma_raw == 10.70 and math.isclose(t["A-fine"].sigma_raw, 10.70 + 1.36)
    assert LR.PROVISIONAL and "A-fine" in LR.PROVISIONAL_NOTE
    assert LR.REFERENCES["Tsige & Grest 2008 (OPLS-AA W&J, 16 Å + tail)"] == 11.25


def test_machine_answers(monkeypatch):
    assert LR.machine_for("yes").gpu and not LR.machine_for("no").gpu and LR.machine_for(True).gpu
    with pytest.raises(ValueError):
        LR.machine_for("maybe")
    from quantabench.md import bench

    monkeypatch.setattr(bench, "find_lammps", lambda refresh=False: (None, "LAMMPS not found"))
    m = LR.machine_for("auto")
    assert not m.gpu and "LAMMPS not found" in m.note
    info = bench.LammpsInfo("lmp", version="4 Jul 2026", packages=["GPU", "KSPACE", "MOLECULE", "OPENMP"],
                            accelerators={"GPU": {"api": "OpenCL"}}, gpu_present=True)
    monkeypatch.setattr(bench, "find_lammps", lambda refresh=False: (info, ""))
    monkeypatch.setattr(LR, "_gpu_check_failed", lambda: False)
    m = LR.detect_machine()
    assert m.gpu and "OpenCL" in m.note
    monkeypatch.setattr(LR, "_gpu_check_failed", lambda: True)
    assert not LR.detect_machine().gpu
    info.gpu_present = False
    monkeypatch.setattr(LR, "_gpu_check_failed", lambda: False)
    assert not LR.detect_machine().gpu
