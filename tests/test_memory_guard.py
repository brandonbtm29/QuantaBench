"""Memory guard for the dxtb steps (chem.memory_guard) and the per-molecule time limit in one process."""

import pytest

from quantabench import resources
from quantabench.chem import engines, memory_guard, pipeline
from quantabench.chem.inputs import from_smiles

PFOA_ANION = [6] * 8 + [9] * 15 + [8] * 2 + [1]        # C8HF15O2 as dxtb counted it: 102 basis functions
TFA = "OC(=O)C(F)(F)F"


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1, gpu="off",
                output_dir=str(tmp_path / "runs"), formats=("xyz",), memory_budget_gb=2.0)
    base.update(kw)
    return pipeline.BuildConfig(**base)


# ------------------------------------------------------------------------------------ the estimate (pure)
def test_basis_size_matches_dxtb_gfn1():
    assert memory_guard.basis_size(PFOA_ANION) == 102          # dxtb's ihelp.nao for PFOA
    assert memory_guard.basis_size([1, 1, 8]) == 8
    assert memory_guard.basis_size([24, 16]) == 18              # Cr, S: s + p + d


def test_scf_factor_by_charge_spin_and_metal():
    assert memory_guard.scf_factor([6, 1], 0, 1) == 1.0
    assert memory_guard.scf_factor([6, 1], -1, 1) == memory_guard.SCF_ANION
    assert memory_guard.scf_factor([6, 1], 2, 1) == memory_guard.SCF_CATION
    assert memory_guard.scf_factor([6, 1], 0, 2) == memory_guard.SCF_DIFFICULT
    assert memory_guard.scf_factor([24, 7, 6], 0, 1) == memory_guard.SCF_DIFFICULT    # Cr


def test_estimate_scales_with_basis_squared_and_batch():
    one = memory_guard.estimate_mb("gfn1-xtb", [6] * 10) - memory_guard.FIXED_MB
    four = memory_guard.estimate_mb("gfn1-xtb", [6] * 20) - memory_guard.FIXED_MB
    assert four == pytest.approx(4 * one)
    b3 = memory_guard.estimate_mb("gfn1-xtb", [6] * 10, batch=3) - memory_guard.FIXED_MB
    assert b3 == pytest.approx(3 * one)
    pol, ef = (memory_guard.estimate_mb(s, PFOA_ANION) - memory_guard.FIXED_MB for s in ("polarizability", "gfn1-xtb"))
    assert pol > 5 * ef
    with pytest.raises(ValueError, match="no memory model"):
        memory_guard.estimate_mb("ccsd", [6])


@pytest.mark.parametrize(("step", "nao", "charge", "measured_mb"), [
    # closed-shell measurements of the module docstring (a fake molecule of carbons with that basis size)
    ("polarizability", 308, 0, 1480), ("polarizability", 246, -1, 1508), ("polarizability", 150, 0, 333),
    ("polarizability", 312, 1, 1551), ("polarizability", 144, -2, 417),
    ("gfn1-xtb", 758, 0, 920), ("gfn1-xtb", 608, 0, 810), ("gfn1-xtb", 246, -1, 237), ("gfn1-xtb", 344, 2, 274),
    ("gfn2-xtb", 458, 0, 1522), ("charges", 758, 0, 122), ("tblite-polarizability", 758, 0, 153),
])
def test_rule_covers_the_measurements(step, nao, charge, measured_mb):
    numbers = [6] * (nao // 4)
    est = memory_guard.estimate_mb(step, numbers, charge)
    assert measured_mb <= est <= 3 * measured_mb


def test_budget_resolution():
    assert memory_guard.resolve_budget_gb(3.0, 6, available=100) == 3.0       # set explicitly
    assert memory_guard.resolve_budget_gb(-1, 6, available=100) == 0.0        # off
    assert memory_guard.resolve_budget_gb(0, 6, available=24) == 4.0          # auto: available / lanes
    assert memory_guard.resolve_budget_gb(0, 12, available=8) == memory_guard.FLOOR_GB   # with a floor


def test_fits_counts_what_the_process_already_uses():
    est = memory_guard.estimate_mb("gfn1-xtb", PFOA_ANION, -1)
    assert memory_guard.fits("gfn1-xtb", PFOA_ANION, -1, 1, 1.0, current_mb=1024 - est - 1)[0]
    ok, why = memory_guard.fits("gfn1-xtb", PFOA_ANION, -1, 1, 1.0, current_mb=1024 - est + 1)
    assert not ok and "102 basis functions, charge -1" in why and "budget 1.0 GB" in why
    assert memory_guard.fits("gfn1-xtb", PFOA_ANION, -1, 1, 0.0, current_mb=1e9) == (True, "")   # guard off


def test_max_batch():
    one = memory_guard.estimate_mb("gfn1-xtb", PFOA_ANION) - memory_guard.FIXED_MB
    room = 4 * one + memory_guard.FIXED_MB
    assert memory_guard.max_batch("gfn1-xtb", PFOA_ANION, 0, 1, (500 + room + 1) / 1024, current_mb=500) == 4
    assert memory_guard.max_batch("gfn1-xtb", PFOA_ANION, 0, 1, 0.5, current_mb=600) == 0
    assert memory_guard.max_batch("gfn1-xtb", PFOA_ANION, 0, 1, 0.0) is None
    assert memory_guard.max_batch("aimnet2", PFOA_ANION, 0, 1, 2.0) is None


# ------------------------------------------------------------------------------- the plan (fake estimates)
@pytest.fixture
def fake_estimate(monkeypatch):
    """Estimates in MB by step (default 1 MB); the process is taken to use 500 MB."""
    sizes: dict[str, float] = {}
    monkeypatch.setattr(memory_guard, "estimate_mb", lambda step, *a, **k: sizes.get(step, 1.0))
    monkeypatch.setattr(memory_guard, "rss_mb", lambda: 500.0)
    return sizes


def test_plan_takes_dxtb_when_it_fits(fake_estimate):
    assert memory_guard.properties_plan(PFOA_ANION, -1, 1, 2.0, tblite=True) == (True, "dxtb", "")


def test_plan_prefers_tblite_for_the_polarizability(fake_estimate):
    fake_estimate["polarizability"] = 5000
    charges, path, note = memory_guard.properties_plan(PFOA_ANION, -1, 1, 2.0, tblite=True)
    assert charges and path == "tblite"
    assert "from tblite GFN1-xTB (finite field) instead of dxtb" in note and "estimated 4.9 GB" in note


def test_plan_skips_the_polarizability_without_tblite(fake_estimate):
    fake_estimate["polarizability"] = 5000
    charges, path, note = memory_guard.properties_plan(PFOA_ANION, -1, 1, 2.0, tblite=False)
    assert charges and path == ""
    assert note.startswith("xTB polarizability skipped by the memory guard: estimated 4.9 GB") and "tblite" in note
    fake_estimate["tblite-polarizability"] = 5000      # tblite would not fit either
    assert memory_guard.properties_plan(PFOA_ANION, -1, 1, 2.0, tblite=True)[1] == ""


def test_plan_skips_everything_when_the_charges_do_not_fit(fake_estimate):
    fake_estimate["charges"] = 5000
    charges, path, note = memory_guard.properties_plan(PFOA_ANION, -1, 1, 2.0, tblite=True)
    assert (charges, path) == (False, "") and note.startswith("xTB properties skipped by the memory guard")


def test_plan_without_a_field_backend_or_guard(fake_estimate):
    fake_estimate["polarizability"] = 5000
    assert memory_guard.properties_plan(PFOA_ANION, 0, 1, 2.0, dxtb_polarizability=False) == (True, "", "")
    assert memory_guard.properties_plan(PFOA_ANION, 0, 1, 0.0) == (True, "dxtb", "")   # budget 0 = off


# ----------------------------------------------------------------------------- in the pipeline (dxtb)
needs_dxtb = pytest.mark.skipif(not engines.available("gfn1-xtb")[0], reason="dxtb/torch not installed")
needs_field = pytest.mark.skipif(not (engines.available("gfn1-xtb")[0] and engines.has_libcint()),
                                 reason="dxtb's libcint backend (field integrals) not installed")


@needs_field
@pytest.mark.skipif(not resources.has_package("tblite"), reason="tblite not installed")
def test_pipeline_takes_the_tblite_polarizability_over_budget(tmp_path, fake_estimate):
    c = cfg(tmp_path, method="gfn1-xtb", xtb_properties=True)
    ref = pipeline.run_molecule(from_smiles(TFA, id="TFA"), c)
    assert ref.status == "ok" and "xtb_polarizability_method" not in ref.descriptors
    fake_estimate["polarizability"] = 5000
    r = pipeline.run_molecule(from_smiles(TFA, id="TFA"), c)
    assert r.status == "ok", r.error
    assert any("from tblite GFN1-xTB (finite field)" in n for n in r.notes)
    assert r.descriptors["xtb_polarizability_method"] == "tblite GFN1-xTB finite field"
    # the same GFN1-xTB quantity; charges and dipole stay dxtb's
    assert r.descriptors["xtb_polarizability_iso_A3"] == pytest.approx(ref.descriptors["xtb_polarizability_iso_A3"],
                                                                       rel=1e-3)
    assert r.descriptors["xtb_dipole_debye"] == pytest.approx(ref.descriptors["xtb_dipole_debye"], abs=1e-6)


@needs_field
def test_pipeline_skips_the_polarizability_with_a_note(tmp_path, fake_estimate, monkeypatch):
    fake_estimate["polarizability"] = 5000
    real = resources.has_package
    monkeypatch.setattr(resources, "has_package", lambda name: False if name == "tblite" else real(name))
    r = pipeline.run_molecule(from_smiles(TFA, id="TFA"), cfg(tmp_path, method="gfn1-xtb", xtb_properties=True))
    assert r.status == "ok", r.error
    assert "xtb_polarizability_iso_A3" not in r.descriptors and "xtb_dipole_debye" in r.descriptors
    notes = [n for n in r.notes if "memory guard" in n]
    assert len(notes) == 1 and notes[0].startswith("xTB polarizability skipped by the memory guard")
    assert "xTB polarizability skipped by the memory guard" in r.summary_row()["notes"]     # -> summary.csv


@needs_dxtb
def test_refinement_over_budget(tmp_path, fake_estimate):
    """An explicit dxtb method fails the molecule with the reason; 'auto' goes on to its next engine."""
    fake_estimate["gfn1-xtb"] = 5000
    r = pipeline.run_molecule(from_smiles("CCO", id="eth"), cfg(tmp_path, method="gfn1-xtb"))
    assert r.status == "failed" and r.error.startswith("MemoryGuardError: GFN1-xTB (dxtb) refinement not started by "
                                                       "the memory guard: estimated 4.9 GB")
    batched = pipeline.run_molecule(from_smiles("CCO", id="eth"), cfg(tmp_path, method="gfn1-xtb", batched="on"))
    assert batched.status == "failed" and "memory guard" in batched.error


@needs_dxtb
def test_batched_refinement_is_capped_by_the_budget(tmp_path, fake_estimate, monkeypatch):
    from quantabench.chem import batch_opt

    sizes = []
    real = batch_opt.optimize_batched

    def spy(model, x0, *a, **k):
        sizes.append(int(x0.shape[0]))
        return real(model, x0, *a, **k)

    monkeypatch.setattr(batch_opt, "optimize_batched", spy)
    monkeypatch.setattr(memory_guard, "max_batch", lambda *a, **k: 2)
    logs = []
    r = pipeline.run_molecule(from_smiles("CCCCO", id="butanol"),
                              cfg(tmp_path, method="gfn1-xtb", batched="on", max_conformers=6, refine_top=5),
                              log=logs.append)
    assert r.status == "ok", r.error
    assert sizes and max(sizes) <= 2
    assert any("memory guard: batches of at most 2 conformer(s)" in m for m in logs)


# --------------------------------------------------------------------------------- time limit, one process
def test_time_limit_stops_at_the_next_checkpoint(tmp_path):
    logs = []
    r = pipeline.run_molecule_limited(from_smiles(TFA, id="TFA"), cfg(tmp_path, molecule_timeout_s=1e-4), "cpu", 1,
                                      lambda: False, logs.append)
    assert r.status == "failed" and r.error.startswith("time limit (0.0001 s) in ")
    assert any("stopped at the next checkpoint" in m for m in logs)


def test_a_cancel_is_not_a_time_limit(tmp_path):
    tl = pipeline.TimeLimit(lambda: True, 1e-4)
    r = pipeline.run_molecule(from_smiles(TFA, id="TFA"), cfg(tmp_path), should_stop=tl)
    assert tl.finish(r).status == "cancelled"


def test_settings_cli_and_gui_get_the_limits():
    """The settings defaults reach BuildConfig.from_settings (the GUI's batch config) and the build options
    override them (cli._config)."""
    from quantabench import cli
    from quantabench.settings import Settings

    s = Settings()
    assert (s.resources.molecule_timeout_s, s.resources.memory_budget_gb) == (1800.0, 0.0)
    s.resources.molecule_timeout_s, s.resources.memory_budget_gb = 600.0, 3.5
    c = pipeline.BuildConfig.from_settings(s)
    assert (c.molecule_timeout_s, c.memory_budget_gb) == (600.0, 3.5)
    args = cli.build_parser().parse_args(["build", "CCO", "--molecule-timeout", "0", "--memory-budget", "-1"])
    c = cli._config(args)
    assert (c.molecule_timeout_s, c.memory_budget_gb) == (0.0, -1.0)
    assert pipeline.memory_budget(c) == 0.0                     # guard off


# ------------------------------------------------------------------------------- frequencies stage (Hessian)
def _fake(n_atoms: int, nao: int) -> list[int]:
    """A molecule of carbons and hydrogens with ``n_atoms`` atoms and ``nao`` GFN1 basis functions."""
    heavy = (nao - 2 * n_atoms) // 2
    return [6] * heavy + [1] * (n_atoms - heavy)


@pytest.mark.parametrize(("step", "method", "n_atoms", "nao", "charge", "measured_mb"), [
    # module docstring, frequencies table
    ("analytic", "gfn1-xtb", 5, 20, 0, 261), ("analytic", "gfn1-xtb", 7, 28, -1, 353),
    ("analytic", "gfn1-xtb", 11, 28, 0, 465), ("analytic", "gfn1-xtb", 11, 44, 0, 647),
    ("analytic", "gfn1-xtb", 13, 38, 0, 675), ("analytic", "gfn1-xtb", 17, 44, 0, 924),
    ("analytic", "gfn1-xtb", 14, 54, 0, 1042), ("analytic", "gfn1-xtb", 13, 52, -1, 1214),
    ("analytic", "gfn2-xtb", 7, 28, -1, 700), ("analytic", "gfn2-xtb", 11, 28, 0, 934),
    ("analytic", "gfn2-xtb", 7, 22, -1, 628),
    ("dipole", "gfn1-xtb", 25, 100, -1, 146), ("dipole", "gfn1-xtb", 53, 158, 0, 240),
    ("dipole", "gfn2-xtb", 53, 158, 0, 344),
])
def test_frequencies_rule_covers_the_measurements(step, method, n_atoms, nao, charge, measured_mb):
    numbers = _fake(n_atoms, nao)
    assert (len(numbers), memory_guard.basis_size(numbers)) == (n_atoms, nao)
    est = memory_guard.estimate_frequencies_mb(step, method, numbers, charge)
    assert measured_mb <= est <= 3 * measured_mb


@pytest.mark.parametrize(("n_atoms", "nao", "charge", "killed_above_mb"), [
    (17, 52, 0, 1360), (23, 68, 0, 1360), (16, 64, -1, 1360), (25, 100, -1, 6400)])
def test_frequencies_rule_catches_the_killed_runs(n_atoms, nao, charge, killed_above_mb):
    assert memory_guard.estimate_frequencies_mb("analytic", "gfn1-xtb", _fake(n_atoms, nao), charge) > killed_above_mb


def test_finite_differences_cost_about_one_force_call():
    for n_atoms, nao, charge, measured in ((13, 52, -1, 22), (25, 100, -1, 45), (37, 148, -1, 74),
                                           (53, 158, 0, 135), (103, 308, 0, 271)):
        est = memory_guard.estimate_frequencies_mb("finite", "gfn1-xtb", _fake(n_atoms, nao), charge)
        assert measured <= est < 1024
    with pytest.raises(ValueError, match="no frequencies memory model"):
        memory_guard.estimate_frequencies_mb("analytic", "aimnet2", [6])
    with pytest.raises(ValueError, match="step must be"):
        memory_guard.estimate_frequencies_mb("numeric", "gfn1-xtb", [6])


def test_frequencies_plan(monkeypatch):
    pfoa = PFOA_ANION
    plan = memory_guard.frequencies_plan("gfn1-xtb", [8, 1, 1], 0, 1, budget_gb=2.0, current_mb=500)
    assert (plan.hessian, plan.ir, plan.notes) == ("analytic", True, [])
    plan = memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, budget_gb=2.0, current_mb=500)
    assert plan.hessian == "finite" and plan.ir
    assert plan.notes[0].startswith("Hessian by finite differences instead of analytic (memory guard: analytic "
                                    "estimated 16.8 GB for 26 atoms")
    plan = memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, hessian="analytic", budget_gb=2.0, current_mb=500)
    assert plan.hessian == "finite"                         # an explicit analytic Hessian too
    plan = memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, budget_gb=0.7, current_mb=500)
    assert plan.hessian == "finite" and not plan.ir and "IR intensities skipped" in plan.notes[-1]
    plan = memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, budget_gb=0.6, current_mb=500)
    assert plan.skipped and plan.notes[0].startswith("frequencies skipped by the memory guard: finite differences")
    plan = memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, hessian="finite", budget_gb=2.0, current_mb=500)
    assert (plan.hessian, plan.notes) == ("finite", [])
    assert memory_guard.frequencies_plan("gfn1-xtb", pfoa, -1, 1, budget_gb=-1).hessian == "analytic"   # off
    # vibrations.analyze(max_analytic_memory_gb=): a cap on the analytic estimate alone
    assert memory_guard.frequencies_plan("gfn1-xtb", [8, 1, 1], budget_gb=-1, analytic_cap_gb=0.0).hessian == "finite"
    assert memory_guard.frequencies_plan("gfn1-xtb", [8, 1, 1], budget_gb=-1, analytic_cap_gb=4.0).hessian == "analytic"


@pytest.fixture
def limits():
    yield resources
    resources.set_molecule_limits()


def test_process_budget_from_the_run(limits, monkeypatch):
    monkeypatch.setattr(memory_guard, "available_gb", lambda: 10.0)
    assert memory_guard.process_budget_gb() == 10.0                     # nothing recorded: auto, one process
    limits.set_molecule_limits(memory_budget_gb=2.5, molecule_timeout_s=60)
    assert memory_guard.process_budget_gb() == 2.5
    limits.set_molecule_limits(pipeline.BuildConfig(memory_budget_gb=-1.0))        # what a lane does
    assert memory_guard.process_budget_gb() == 0.0
    assert limits.molecule_limits().molecule_timeout_s == pipeline.BuildConfig().molecule_timeout_s


def test_analyze_takes_finite_differences_over_budget(monkeypatch):
    pytest.importorskip("dxtb")
    from ase.build import molecule

    from quantabench.chem import vibrations

    monkeypatch.setattr(memory_guard, "rss_mb", lambda: 0.0)
    water = molecule("H2O")
    # water: analytic ~255 MB, finite differences and dipole derivatives ~31 MB each
    vr = vibrations.analyze(water, "gfn1-xtb", hessian="auto", memory_budget_gb=0.1, symmetry_number=2)
    assert vr.hessian_method == "finite" and vr.n_modes == 3
    assert any(w.startswith("Hessian by finite differences instead of analytic (memory guard") for w in vr.warnings)
    with pytest.raises(memory_guard.MemoryGuardError, match="frequencies skipped by the memory guard"):
        vibrations.analyze(water, "gfn1-xtb", memory_budget_gb=0.02, symmetry_number=2)
