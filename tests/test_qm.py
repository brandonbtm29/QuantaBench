"""QM stage (chem.qm + chem.psi4_driver): level parsing, task building, the driver's numerics, result parsing
with a fake Psi4 interpreter, and real Psi4 runs on water and HF (marked ``external``)."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest

from quantabench import settings as settings_mod
from quantabench.chem import external, psi4_driver, qm
from quantabench.chem.optimize import Cancelled


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(settings_mod.ENV_SETTINGS_DIR, str(tmp_path / "settings"))
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    s.engines.qm_memory_gb = 1.0
    settings_mod.save(s)
    external.clear_cache()
    yield
    external.clear_cache()


def _result(smiles: str = "O", n_conf: int = 2, charge: int = 0, multiplicity: int = 1, id: str = "water"):
    """A MoleculeResult with MMFF conformers, as the pipeline hands it to the QM stage."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from quantabench.chem.conformers import ConformerSet
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import ConformerResult, MoleculeResult

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, n_conf, randomSeed=11))
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    res = MoleculeResult(input=from_smiles(smiles, id=id), charge=charge, multiplicity=multiplicity, mol=mol,
                         method="mmff")
    res.conformers = [ConformerResult(rank=i, conf_id=c, energy_ev=0.1 * i) for i, c in enumerate(ids)]
    return res, ConformerSet(mol, ids, [0.0] * len(ids), "MMFF94s", len(ids))


def _cfg(**kw):
    from quantabench.chem.pipeline import BuildConfig

    base = dict(method="mmff", qm_method="hf/def2-svp", qm_top=2, write_files=False, xtb_properties=False)
    base.update(kw)
    return BuildConfig(**base)


# ----------------------------------------------------------------------------- level and task
@pytest.mark.parametrize("level, expected", [
    ("wB97X-D/def2-SVP", ("wb97x-d", "def2-svp")),
    ("ωB97X-D/6-31G(d)", ("wb97x-d", "6-31g*")),
    ("b3lyp-d3(bj)/def2tzvp", ("b3lyp-d3bj", "def2-tzvp")),
    (" HF / 6-31G* ", ("hf", "6-31g*")),
    ("rhf/aug-cc-pvdz", ("hf", "aug-cc-pvdz")),
])
def test_parse_level(level, expected):
    assert qm.parse_level(level) == expected


@pytest.mark.parametrize("level, match", [("b3lyp", "method/basis"), ("", "method/basis"), ("a/b/c", "method/basis"),
                                          ("/def2-svp", "lacks a method"), ("b3lyp/", "lacks a basis"),
                                          ("b3lyp/def2;rm", "unexpected")])
def test_parse_level_errors(level, match):
    with pytest.raises(ValueError, match=match):
        qm.parse_level(level)


def test_build_task_lowest_conformers_and_settings():
    res, cs = _result(n_conf=3)
    warnings: list[str] = []
    task = qm.build_task(res, cs, _cfg(qm_method="wB97X-D/def2-SVP", qm_top=2, qm_task="optimize", qm_cubes=True,
                                       qm_frequencies=True, solvent="water"), threads=3, warnings=warnings)
    assert task["symbols"] == ["O", "H", "H"]
    assert [c["id"] for c in task["conformers"]] == [res.conformers[0].conf_id, res.conformers[1].conf_id]
    pos = res.mol.GetConformer(res.conformers[1].conf_id).GetPositions()
    assert np.allclose(task["conformers"][1]["positions_A"], pos, atol=1e-7)
    assert (task["method"], task["basis"], task["task"]) == ("wb97x-d", "def2-svp", "optimize")
    assert task["threads"] == 3 and task["memory_gb"] == 1.0
    assert task["properties"] and task["polarizability"] and task["cubes"] and task["hessian"]
    assert task["solvent_pcm"] == "water" and not warnings
    assert psi4_driver.validate_task(json.loads(json.dumps(task)))["conformers"][0]["id"] == task["conformers"][0]["id"]
    assert psi4_driver.reference_for(task["method"], task["multiplicity"]) == "rks"


def test_build_task_solvent_without_pcm_parameters_and_bad_task():
    res, cs = _result(n_conf=1)
    warnings: list[str] = []
    task = qm.build_task(res, cs, _cfg(solvent="octanol", qm_properties=False), warnings=warnings)
    assert task["solvent_pcm"] is None and any("gas phase" in w for w in warnings)
    assert not task["polarizability"]
    with pytest.raises(ValueError, match="qm_task"):
        qm.build_task(res, cs, _cfg(qm_task="freq"))
    with pytest.raises(ValueError, match="only psi4"):
        qm.build_task(res, cs, _cfg(qm_program="orca"))


# ----------------------------------------------------------------------------- driver numerics (no psi4)
def test_driver_validate_task_errors():
    good = {"symbols": ["H", "H"], "positions_A": [[0, 0, 0], [0, 0, 0.74]], "method": "HF", "basis": "def2-SVP"}
    t = psi4_driver.validate_task(good)
    assert t["conformers"][0]["id"] == 0 and t["method"] == "hf" and t["task"] == "energy" and t["threads"] == 1
    for bad, match in (({**good, "positions_A": [[0, 0, 0]]}, "2 x 3"), ({**good, "method": ""}, "method"),
                       ({**good, "task": "md"}, "task"), ({k: v for k, v in good.items() if k != "positions_A"},
                                                          "positions_A"), ({**good, "multiplicity": 0}, "multiplicity")):
        with pytest.raises(psi4_driver.TaskError, match=match):
            psi4_driver.validate_task(bad)
    assert psi4_driver.reference_for("hf", 2) == "uhf" and psi4_driver.reference_for("b3lyp", 3) == "uks"
    assert psi4_driver.reference_for("hf", 1) == "rhf"


def test_driver_polarizability_from_finite_field_dipoles():
    alpha = np.array([[3.0, 0.1, 0.0], [0.1, 6.8, 0.0], [0.0, 0.0, 5.0]])
    f = 0.002
    mu0 = np.array([0.0, 0.0, -0.84])
    # Psi4's perturbation lowers the dipole along +F (mu = mu0 - alpha F): the sign is fixed by the trace
    plus = np.array([mu0 - alpha[:, k] * f for k in range(3)])
    minus = np.array([mu0 + alpha[:, k] * f for k in range(3)])
    got, sign = psi4_driver.alpha_from_dipoles(plus, minus, f)
    assert sign == -1 and np.allclose(got, alpha)
    got2, sign2 = psi4_driver.alpha_from_dipoles(minus, plus, f)
    assert sign2 == 1 and np.allclose(got2, alpha)
    s = psi4_driver.polarizability_summary(alpha)
    eig = np.linalg.eigvalsh(alpha)
    assert s["iso_au"] == pytest.approx(np.trace(alpha) / 3)
    assert s["iso_A3"] == pytest.approx(np.trace(alpha) / 3 * 0.1481847)
    assert s["aniso_A3"] == pytest.approx((eig[-1] - eig[0]) * 0.1481847)
    assert 0 < s["aniso_invariant_au"] <= s["aniso_au"]


def test_driver_homo_lumo_and_cube_boundary(tmp_path):
    eps = [-20.0, -1.3, -0.7, -0.5, 0.2, 0.3]
    assert psi4_driver.homo_lumo(eps, eps, 4, 4, True) == (-0.5, 0.2)
    # open shell: alpha has one more electron; HOMO is the highest occupied of either spin
    assert psi4_driver.homo_lumo(eps, [-19.0, -1.2, -0.6, 0.1, 0.4, 0.5], 4, 3, False) == (-0.5, 0.1)
    data = np.zeros((4, 5, 6))
    data[2, 2, 3] = 0.3
    data[0, 2, 2] = 0.001
    lines = ["cube", "comment", "    1    0.000000    0.000000    0.000000",
             "    4    0.200000    0.000000    0.000000", "    5    0.000000    0.200000    0.000000",
             "    6    0.000000    0.000000    0.200000", "    8    0.000000    0.000000    0.000000    0.000000"]
    vals = data.ravel()
    lines += [" ".join(f"{v:.5E}" for v in vals[i:i + 6]) for i in range(0, vals.size, 6)]
    path = tmp_path / "Dt.cube"
    path.write_text("\n".join(lines) + "\n")
    shape, back = psi4_driver.read_cube_header_and_data(path)
    assert shape == (4, 5, 6) and np.allclose(back, data)
    assert psi4_driver.boundary_max(back) == pytest.approx(0.001)


# ----------------------------------------------------------------------------- fake Psi4
# HF/def2-SVP water from Psi4 1.9.1 (optimized): Hessian in Eh/bohr², dipole derivatives (3N x 3) in e,
# and Psi4's own frequencies (cm-1) and IR intensities (km/mol) for comparison.
WATER_HF = {
    "positions_A": [[0.0, 0.0, 0.11057452], [0.0, 0.75036514, -0.46378726], [0.0, -0.75036514, -0.46378726]],
    "energy_hartree": -75.96128289156457,
    "hessian_au": [
        [-7.4e-07, 0.0, 0.0, 3.7e-07, 0.0, 0.0, 3.7e-07, -0.0, -0.0],
        [0.0, 0.80297643, -0.0, 0.0, -0.40148822, 0.30731665, -0.0, -0.40148822, -0.30731665],
        [0.0, -0.0, 0.54244154, -0.0, 0.24244609, -0.27122077, -0.0, -0.24244609, -0.27122077],
        [3.7e-07, 0.0, -0.0, -4.8e-07, -0.0, -0.0, 1.1e-07, -0.0, 0.0],
        [0.0, -0.40148822, 0.24244609, -0.0, 0.43476704, -0.27488137, -0.0, -0.03327882, 0.03243528],
        [0.0, 0.30731665, -0.27122077, -0.0, -0.27488137, 0.25322679, -0.0, -0.03243528, 0.01799398],
        [3.7e-07, -0.0, -0.0, 1.1e-07, -0.0, -0.0, -4.8e-07, 0.0, 0.0],
        [-0.0, -0.40148822, -0.24244609, -0.0, -0.03327882, -0.03243528, 0.0, 0.43476704, 0.27488137],
        [-0.0, -0.30731665, -0.27122077, 0.0, 0.03243528, 0.01799398, 0.0, 0.27488137, 0.25322679]],
    "dipole_derivatives_au": [
        [-0.7647753, 0.0, 0.0], [-0.0, -0.56167975, 0.0], [0.0, 0.0, -0.44547553],
        [0.38238768, -0.0, -0.0], [0.0, 0.28083987, 0.02732568], [-0.0, 0.07772908, 0.22273776],
        [0.38238768, -0.0, -0.0], [-0.0, 0.28083987, -0.02732568], [-0.0, -0.07772908, 0.22273776]],
    "gradient_au": [[0.0, 0.0, -1e-06], [0.0, -1e-06, 0.0], [0.0, 1e-06, 0.0]],
    "hessian_kind": "analytic", "optimized": True, "opt_converged": True,
    "psi4_frequencies_cm1": [1750.776, 4148.986, 4245.121], "psi4_ir_km_mol": [82.7436, 26.7769, 74.4865],
}

FAKE_DRIVER = r'''
import json, os, sys, time
from pathlib import Path

MODE = "__MODE__"
FREQ = json.loads(__FREQ__)
task = json.loads(Path(sys.argv[-1]).read_text())
if os.environ.get("QB_FAKE_TASK_COPY"):
    Path(os.environ["QB_FAKE_TASK_COPY"]).write_text(json.dumps({"argv": sys.argv, "task": task,
        "threads": os.environ.get("OMP_NUM_THREADS"), "scratch": os.environ.get("PSI_SCRATCH")}))
print("QB| fake Psi4 starting", flush=True)
if MODE == "slow":
    time.sleep(60)
if MODE == "crash":
    print("Segmentation fault (core dumped)")
    sys.exit(139)
if MODE == "fail":
    Path("result.json").write_text(json.dumps({"ok": False, "error": "SCF did not converge in 200 iterations"}))
    sys.exit(1)
n = len(task["conformers"])
energies = [-75.95 - 0.01 * i for i in range(n)]          # the last conformer is the lowest
Path("output.dat").write_text("fake psi4 output\n")
out = {"ok": True, "program": "psi4", "program_version": "1.9.1", "level": task["method"] + "/" + task["basis"],
       "reference": "rhf", "task": task["task"], "solvent": task.get("solvent") or "", "warnings": ["fake warning"],
       "timings": {"total": 0.1}, "conformers": []}
for i, (c, e) in enumerate(zip(task["conformers"], energies)):
    rec = {"id": c["id"], "energy_hartree": e, "converged": True, "error": "", "seconds": 0.01}
    if task["task"] == "optimize":
        rec["positions_A"] = c["positions_A"]
    out["conformers"].append(rec)
    print(f"QB| conformer {i + 1}/{n} (id {c['id']}): E = {e:.8f} Eh", flush=True)
best = n - 1
out.update(best_index=best, best_id=task["conformers"][best]["id"], energy_hartree=energies[best],
           best_positions_A=task["conformers"][best]["positions_A"])
if task["properties"]:
    out["properties"] = {"dipole_debye": 2.13, "dipole_vector_debye": [0.0, 0.0, -2.13],
                         "quadrupole_debye_A": [[-7.0, 0, 0], [0, -4.1, 0], [0, 0, -5.7]], "homo_ev": -13.55,
                         "lumo_ev": 4.80, "gap_ev": 18.35,
                         "charges": {"mulliken": [-0.35, 0.175, 0.175], "lowdin": [-0.15, 0.075, 0.075],
                                     "mbis": [-0.87, 0.435, 0.435]},
                         "mbis_volumes_bohr3": [26.2, 1.34, 1.34], "mbis_volume_ratios": [1.33, 0.18, 0.18]}
if task["polarizability"]:
    out["polarizability"] = {"iso_A3": 0.7265, "aniso_A3": 0.562, "aniso_invariant_A3": 0.487,
                             "tensor_A3": [[0.44, 0, 0], [0, 1.0, 0], [0, 0, 0.73]],
                             "method": "analytic (CPSCF linear response)"}
if task["cubes"]:
    Path("cubes").mkdir(exist_ok=True)
    Path("cubes/Dt.cube").write_text("density cube\n")
    Path("cubes/ESP.cube").write_text("esp cube\n")
    out["cubes"] = {"density": "cubes/Dt.cube", "esp": "cubes/ESP.cube", "shape": [2, 2, 2], "spacing_bohr": 0.2,
                    "overage_bohr": 4.0, "boundary_density_max_au": 1e-4, "n_points": 8}
if task["hessian"]:
    out["frequencies"] = FREQ
Path("result.json").write_text(json.dumps(out))
if MODE == "hang":            # Psi4 stuck at exit after a failed SCF left its timers on
    time.sleep(60)
'''


def _fake_python(tmp_path: Path, monkeypatch, mode: str = "ok") -> Path:
    """An executable that behaves like ``<psi4 python> -I psi4_driver.py task.json`` (Windows and Linux)."""
    script = tmp_path / f"fake_psi4_{mode}.py"
    body = FAKE_DRIVER.replace("__MODE__", mode).replace("__FREQ__", repr(json.dumps(WATER_HF)))
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        exe = tmp_path / f"python_{mode}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = tmp_path / f"python_{mode}"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    monkeypatch.setenv("QUANTABENCH_PSI4_PYTHON", str(exe))
    return exe


def test_run_stage_with_fake_psi4(tmp_path, monkeypatch):
    from quantabench.chem import surfaces

    _fake_python(tmp_path, monkeypatch)
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("QB_FAKE_TASK_COPY", str(seen))
    calls = []

    def fake_surface(density, esp=None, iso=0.002):
        calls.append((density, esp))
        return {"surf_area_A2": 40.0, "surf_volume_A3": 21.0, "esp_min_kcal": -35.0}

    monkeypatch.setattr(surfaces, "surface_descriptors", fake_surface)
    res, cs = _result(n_conf=2)
    lines: list[str] = []
    cfg = _cfg(qm_top=2, qm_cubes=True, qm_frequencies=True, temperature_k=298.15)
    qm.run_stage(res, cs, cfg, threads=2, log=lines.append)

    sent = json.loads(seen.read_text())
    assert sent["argv"][-2:] == ["psi4_driver.py", "task.json"] and sent["threads"] == "2"
    assert Path(sent["scratch"]).parent == tmp_path / "scratch"        # PSI_SCRATCH = the job folder
    assert not Path(sent["scratch"]).exists()                          # removed after success
    q = res.qm
    assert set(qm.QM_RESULT_KEYS) <= set(q)
    assert q["level"] == "hf/def2-svp" and q["program_version"] == "1.9.1" and q["task"] == "energy"
    ids = [c.conf_id for c in res.conformers]
    assert [c["conf_id"] for c in q["conformers"]] == ids and q["best_conf_id"] == ids[1]
    assert q["energy_ev"] == pytest.approx(-75.96 * 27.211386245988)
    assert q["conformers"][0]["rel_kcal"] == pytest.approx(0.01 * 27.211386245988 * 23.060548, rel=1e-6)
    assert q["properties"]["polarizability_iso_A3"] == 0.7265 and q["properties"]["conf_id"] == ids[1]
    for label in ("log", "density", "esp"):
        assert Path(q["files"][label]).is_file(), label
    assert calls == [(q["files"]["density"], q["files"]["esp"])]
    d = res.descriptors
    assert d["qm_level"] == "hf/def2-svp" and d["qm_dipole_debye"] == 2.13 and d["qm_gap_ev"] == 18.35
    assert d["qm_polarizability_iso_A3"] == 0.7265 and d["qm_mbis_charge_min"] == -0.87
    assert d["qm_surf_area_A2"] == 40.0 and d["qm_surf_esp_min_kcal"] == -35.0
    # vibrations: normal modes and IR from the Psi4 Hessian reproduce Psi4's own analysis
    v = q["vibrations"]
    assert np.allclose(v["frequencies_cm1"], WATER_HF["psi4_frequencies_cm1"], atol=0.5)
    assert np.allclose(v["ir_intensities_km_mol"], WATER_HF["psi4_ir_km_mol"], rtol=0.02)
    assert v["hessian_method"] == "analytic" and v["thermo"]["symmetry_number"] == 2
    assert v["thermo"]["zpe_kcal"] == pytest.approx(sum(WATER_HF["psi4_frequencies_cm1"]) / 2 / 349.755, rel=1e-3)
    assert len(v["displacements"]) == 3 and v["symbols"] == ["O", "H", "H"] and v["n_imaginary"] == 0
    assert d["qm_zpe_kcal"] == v["thermo"]["zpe_kcal"] and d["qm_symmetry_number"] == 2
    assert "QM: fake warning" in res.notes
    assert any("QM: conformer 2/2" in line for line in lines) and any("QM done" in line for line in lines)
    json.dumps(q)       # JSON-ready


def test_pipeline_and_export_with_fake_psi4(tmp_path, monkeypatch):
    from quantabench.chem import export
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import run_molecule

    _fake_python(tmp_path, monkeypatch)
    cfg = _cfg(qm_top=1, qm_cubes=True, qm_frequencies=True, qm_task="optimize", max_conformers=2)
    res = run_molecule(from_smiles("O", id="water"), cfg)
    assert res.status == "ok", res.error
    assert res.qm is not None and not any("QM stage skipped" in n for n in res.notes)
    assert any("surface descriptors" in w for w in res.qm["warnings"]) or res.qm["surface"] is not None
    files = export.write_molecule(res, tmp_path / "out", ("xyz", "spectrum"))
    names = {Path(f).name for f in files}
    assert {"water_qm_log.out", "water_qm_density.cube", "water_qm_esp.cube", "water_qm_geometry.xyz",
            "water_qm_ir.csv", "water_qm_modes.csv", "water_qm_freq.xyz"} <= names
    assert res.qm["files"]["density"] == str(tmp_path / "out" / "water_qm_density.cube")


def test_surface_not_implemented_is_a_warning(tmp_path, monkeypatch):
    from quantabench.chem import surfaces

    def not_yet(*a, **k):
        raise NotImplementedError("later")

    monkeypatch.setattr(surfaces, "surface_descriptors", not_yet)
    _fake_python(tmp_path, monkeypatch)
    res, cs = _result(n_conf=1)
    qm.run_stage(res, cs, _cfg(qm_top=1, qm_cubes=True))
    assert res.qm["surface"] is None and any("not available yet" in w for w in res.qm["warnings"])
    assert not any(k.startswith("qm_surf") for k in res.descriptors)


def test_driver_failure_is_reported(tmp_path, monkeypatch):
    _fake_python(tmp_path, monkeypatch, "fail")
    res, cs = _result(n_conf=1)
    with pytest.raises(external.ExternalProgramError, match="SCF did not converge") as err:
        qm.run_stage(res, cs, _cfg())
    assert "job files kept in" in str(err.value)
    assert res.qm is None


def test_driver_crash_without_result(tmp_path, monkeypatch):
    _fake_python(tmp_path, monkeypatch, "crash")
    res, cs = _result(n_conf=1)
    with pytest.raises(external.ExternalProgramError, match="without a result") as err:
        qm.run_stage(res, cs, _cfg())
    assert "Segmentation fault" in str(err.value)


def test_cancel_kills_driver(tmp_path, monkeypatch):
    _fake_python(tmp_path, monkeypatch, "slow")
    res, cs = _result(n_conf=1)
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    with pytest.raises(Cancelled):
        qm.run_stage(res, cs, _cfg(), should_stop=flag.is_set)


def test_driver_that_does_not_exit_after_its_result(tmp_path, monkeypatch):
    import time

    _fake_python(tmp_path, monkeypatch, "hang")
    monkeypatch.setattr(qm, "EXIT_GRACE_S", 0.5)
    res, cs = _result(n_conf=1)
    t0 = time.perf_counter()
    qm.run_stage(res, cs, _cfg())
    assert time.perf_counter() - t0 < 30                     # not the fake's 60 s
    assert res.qm["energy_ev"] is not None and any("did not exit" in w for w in res.qm["warnings"])
    # a real cancellation is still a cancellation
    flag = threading.Event()
    threading.Timer(0.3, flag.set).start()
    _fake_python(tmp_path, monkeypatch, "slow")
    with pytest.raises(Cancelled):
        qm.run_stage(res, cs, _cfg(), should_stop=flag.is_set)


def test_free_atom_scf_failure_cleans_psi4_scratch():
    """A failed atomic SCF must not leave its scratch files (DIIS) to the next SCF."""
    from types import SimpleNamespace

    calls: list[str] = []

    class Wfn:
        def __init__(self, sym):
            self.sym = sym

        def molecule(self):
            return SimpleNamespace(natom=lambda: 2, symbol=lambda i: ("C", "F")[i], Z=lambda i: (6, 9)[i])

        def variable(self, name):
            return [12.5]

        def set_variable(self, name, value):
            calls.append(f"set {name}")

    def energy(method, molecule, return_wfn):
        sym = molecule.elem
        calls.append(f"energy {sym}")
        if sym == "F":
            raise RuntimeError("Could not converge SCF iterations in 200 iterations.")
        return -1.0, Wfn(sym)

    core = SimpleNamespace(
        Molecule=SimpleNamespace(from_arrays=lambda elem, **kw: SimpleNamespace(elem=elem[0], update_geometry=lambda: None)),
        clean=lambda: calls.append("clean"), set_active_molecule=lambda m: None)
    fake = SimpleNamespace(core=core, energy=energy, oeprop=lambda *a, **k: None, set_options=lambda o: None)
    t = {"method": "wb97x-d", "multiplicity": 1}
    with pytest.raises(RuntimeError, match="converge"):
        psi4_driver.free_atom_volumes(fake, t, Wfn("mol"), pcm=False)
    assert calls == ["energy C", "clean", "set MBIS FREE ATOM C VOLUME", "energy F", "clean"]


def test_missing_psi4(monkeypatch):
    monkeypatch.setattr(external, "find_python_with", lambda *a, **k: None)
    assert qm.available()[0] is False
    res, cs = _result(n_conf=1)
    with pytest.raises(external.ExternalProgramError, match="Psi4 not found"):
        qm.run_stage(res, cs, _cfg())


# ----------------------------------------------------------------------------- real Psi4
@pytest.fixture
def psi4_python():
    py = external.find_python_with("psi4", "psi4_python")
    if py is None:
        pytest.skip("Psi4 not installed (set QUANTABENCH_PSI4_PYTHON)")
    return py


def _driver(task: dict, wd: Path, python: str) -> dict:
    return qm.run_driver(psi4_driver.validate_task(task), wd, python, threads=1)


WATER = {"symbols": ["O", "H", "H"], "positions_A": [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]],
         "method": "hf", "basis": "def2-svp", "threads": 1, "memory_gb": 1.0}


@pytest.mark.external
def test_psi4_water_hf_properties_and_cubes(psi4_python):
    res, cs = _result(n_conf=2)
    lines: list[str] = []
    qm.run_stage(res, cs, _cfg(qm_top=2, qm_cubes=True), threads=1, log=lines.append)
    q = res.qm
    assert q["program"] == "psi4" and q["level"] == "hf/def2-svp"
    assert all(c["converged"] and -76.0 < c["energy_hartree"] < -75.9 for c in q["conformers"])
    p = q["properties"]
    assert 1.8 < p["dipole_debye"] < 2.5 and p["homo_ev"] < 0 < p["lumo_ev"]
    assert 0.6 < p["polarizability_iso_A3"] < 0.9 and p["polarizability_method"].startswith("analytic")
    for kind in ("mulliken", "lowdin", "mbis"):
        assert sum(p["charges"][kind]) == pytest.approx(0.0, abs=1e-3), kind
    assert p["charges"]["mbis"][0] < -0.5 and len(p["mbis_volumes_bohr3"]) == 3
    assert q["cubes"]["boundary_density_max_au"] < 0.002
    assert all(Path(q["files"][k]).stat().st_size > 1000 for k in ("density", "esp", "log"))
    assert res.descriptors["qm_homo_ev"] == p["homo_ev"]
    assert any("QM: conformer 2/2" in line for line in lines)


@pytest.mark.external
def test_psi4_polarizability_analytic_matches_finite_field(psi4_python, tmp_path):
    base = {**WATER, "properties": False, "polarizability": True}
    ana = _driver({**base, "polarizability_method": "analytic"}, tmp_path / "a", psi4_python)["polarizability"]
    ff = _driver({**base, "polarizability_method": "finite_field"}, tmp_path / "f", psi4_python)["polarizability"]
    assert ana["method"].startswith("analytic") and ff["method"].startswith("finite field")
    assert ff["iso_au"] == pytest.approx(ana["iso_au"], rel=0.01)
    assert np.allclose(ff["tensor_au"], ana["tensor_au"], atol=0.02)
    assert ana["iso_au"] == pytest.approx(4.9025, rel=0.01)     # HF/def2-SVP at this geometry


@pytest.mark.external
def test_psi4_hydrogen_fluoride_opt_freq_analytic(psi4_python):
    res, cs = _result("F", n_conf=1, id="hf")
    qm.run_stage(res, cs, _cfg(qm_top=1, qm_task="optimize", qm_frequencies=True, qm_properties=False), threads=1)
    v = res.qm["vibrations"]
    assert v["geometry"] == "linear" and v["n_modes"] == 1 and v["hessian_method"] == "analytic"
    assert 4300 < v["frequencies_cm1"][0] < 4700
    assert v["frequencies_cm1"][0] == pytest.approx(v["psi4_frequencies_cm1"][0], abs=0.5)
    assert v["ir_intensities_km_mol"][0] > 50 and v["thermo"]["symmetry_number"] == 1
    assert res.qm["conformers"][0]["positions_A"] and Path(res.qm["files"]["geometry"]).is_file()


@pytest.mark.external
def test_psi4_water_wb97xd_frequencies_findif(psi4_python):
    res, cs = _result(n_conf=1)
    qm.run_stage(res, cs, _cfg(qm_method="wb97x-d/def2-svp", qm_top=1, qm_frequencies=True, qm_properties=False),
                 threads=1)
    v = res.qm["vibrations"]
    nu = v["frequencies_cm1"]
    assert 1600 < nu[0] < 1680 and 3850 < nu[1] < 3950 and 3950 < nu[2] < 4050
    assert np.allclose(nu, v["psi4_frequencies_cm1"], atol=1.0)
    assert v["hessian_kind"].startswith("finite differences") and v["ir_intensities_km_mol"] is not None
    assert np.allclose(v["ir_intensities_km_mol"], v["psi4_ir_km_mol"], rtol=0.05, atol=0.5)
    assert res.descriptors["qm_zpe_kcal"] == pytest.approx(sum(nu) / 2 / 349.755, rel=1e-3)   # ~13.6 kcal/mol


@pytest.mark.external
def test_psi4_pcm_and_open_shell(psi4_python, tmp_path):
    gas = _driver(WATER, tmp_path / "gas", psi4_python)
    pcm = _driver({**WATER, "solvent_pcm": "water"}, tmp_path / "pcm", psi4_python)
    assert pcm["solvent"] == "water" and pcm["energy_hartree"] < gas["energy_hartree"] - 0.003
    oh = _driver({"symbols": ["O", "H"], "positions_A": [[0, 0, 0], [0, 0, 0.97]], "multiplicity": 2,
                  "method": "hf", "basis": "def2-svp", "properties": True, "threads": 1, "memory_gb": 1.0},
                 tmp_path / "oh", psi4_python)
    assert oh["reference"] == "uhf" and oh["properties"]["n_alpha"] == oh["properties"]["n_beta"] + 1
    assert oh["properties"]["homo_ev"] < oh["properties"]["lumo_ev"]


@pytest.mark.external
def test_psi4_bad_level_is_a_clear_error(psi4_python, tmp_path):
    with pytest.raises(external.ExternalProgramError, match="basis set 'no-such-basis'"):
        _driver({**WATER, "basis": "no-such-basis"}, tmp_path / "b", psi4_python)
    with pytest.raises(external.ExternalProgramError, match="not a Psi4 method"):
        _driver({**WATER, "method": "b3lpy"}, tmp_path / "m", psi4_python)


def test_export_writes_utf8_names_whatever_the_locale(tmp_path):
    """On Windows ASE opens a path in cp1252 and a molecule named "β-..." failed the export; the files are
    written as UTF-8 now. A child Python with an ASCII locale stands in for Windows here."""
    import os
    import subprocess
    import sys

    code = """
import sys
from pathlib import Path
from quantabench.chem import export, pipeline
from quantabench.chem.inputs import from_smiles
cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1,
                           write_files=False)
res = pipeline.run_molecule(from_smiles("OCC(F)(F)F", id="beta-T", name="\\u0392-Tocopherol test"), cfg)
assert res.status == "ok", res.error
files = export.write_molecule(res, Path(sys.argv[1]), ("xyz", "xyz-all", "extxyz", "sdf"))
print(sys.getfilesystemencoding(), open(files[0], "rb").read().decode("utf-8").splitlines()[1][:20])
"""
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
           "PYTHONIOENCODING": "utf-8"}
    probe = subprocess.run([sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
                           env=env, capture_output=True, text=True)
    if probe.stdout.strip().lower().replace("-", "") in ("utf8",):
        pytest.skip("cannot get a non-UTF-8 locale here")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "out")], env=env, capture_output=True,
                         encoding="utf-8", timeout=300)      # the child prints UTF-8 (PYTHONIOENCODING)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Β-Tocopherol test" in out.stdout
