import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from conftest import PFOA

from quantabench.chem import engines, pipeline
from quantabench.chem.inputs import from_smiles


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=4, refine_top=2, xtb_properties=False, output_dir=str(tmp_path / "runs"),
                formats=("xyz", "sdf", "extxyz", "lammps", "pdb", "xyz-all", "mol"))
    base.update(kw)
    return pipeline.BuildConfig(**base)


def test_mmff_end_to_end(tmp_path):
    results, run_dir = pipeline.run([from_smiles(PFOA, id="PFOA")], cfg(tmp_path))
    r = results[0]
    assert r.status == "ok", r.error
    assert r.charge == -1 and r.method == "mmff"
    assert abs(sum(c.weight for c in r.conformers) - 1) < 1e-9
    d = r.descriptors
    assert d["n_cf2"] == 6 and d["n_cf3"] == 1 and d["n_fluorine"] == 15
    assert d["volume_A3_best"] > 100 and d["sasa_A2_best"] > 100
    names = {p.split("/")[-1].split("\\")[-1] for p in r.files}
    assert {"PFOA.xyz", "PFOA_conformers.sdf", "PFOA.lammps.data", "PFOA_summary.json"} <= names
    assert (run_dir / "summary.csv").exists() and (run_dir / "descriptors.csv").exists()


def test_failure_is_reported_not_raised(tmp_path):
    bad = from_smiles("CCO", id="ok")
    bad.smiles = "C1CC"  # unclosed ring
    results, _ = pipeline.run([bad], cfg(tmp_path))
    assert results[0].status == "failed" and results[0].error


class Harmonic(Calculator):
    """Toy engine: pulls every atom toward its starting position; fails on non-CPU devices."""

    implemented_properties = ["energy", "forces", "charges"]

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ref = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if self.device != "cpu":
            raise RuntimeError("no GPU kernels for this device")
        super().calculate(atoms, properties, system_changes)
        if self.ref is None:
            self.ref = atoms.positions.copy() + 0.05
        d = atoms.positions - self.ref
        self.results = {"energy": float((d**2).sum()), "forces": -2 * d,
                        "charges": np.zeros(len(atoms))}


def test_engine_gpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    monkeypatch.setattr(engines, "supports", lambda key, numbers: (True, ""))
    monkeypatch.setattr(engines, "make_calculator", lambda key, numbers, q, m, dev: Harmonic(dev))
    inp = from_smiles(PFOA, id="PFOA")
    r = pipeline.run_molecule(inp, cfg(tmp_path, method="gfn1-xtb"), device="cuda:0", threads=1)
    assert r.status == "ok", r.error
    assert r.device == "cpu" and any("fell back to CPU" in n for n in r.notes)
    assert r.conformers[0].charges is not None and r.conformers[0].dipole_debye is not None


def test_cancel(tmp_path):
    results, _ = pipeline.run([from_smiles(PFOA, id="a"), from_smiles("CCO", id="b")], cfg(tmp_path),
                              should_stop=lambda: True)
    assert results == []


@pytest.mark.skipif(not engines.available("gfn1-xtb")[0], reason="dxtb/torch not installed")
def test_gfn1_xtb_real(tmp_path):
    r = pipeline.run_molecule(from_smiles("OC(=O)C(F)(F)F", id="TFA"),
                              cfg(tmp_path, method="gfn1-xtb", xtb_properties=True), device="cpu", threads=1)
    assert r.status == "ok", r.error
    assert r.descriptors["xtb_polarizability_iso_A3"] > 1
    assert abs(sum(r.conformers[0].charges) + 1) < 1e-6


def test_auto_falls_through_when_engine_cannot_start(tmp_path, monkeypatch):
    real_available = engines.available
    monkeypatch.setattr(engines, "available", lambda key: (True, "") if key == "aimnet2" else
                        (False, "x") if key in ("gfn1-xtb", "gfn2-xtb") else real_available(key))

    def boom(*_a, **_k):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(engines, "make_calculator", boom)
    r = pipeline.run_molecule(from_smiles("CCO", id="eth"), cfg(tmp_path, method="auto"), device="cpu", threads=1)
    assert r.status == "ok", r.error
    assert r.method == "mmff" and any("model download failed" in n for n in r.notes)
