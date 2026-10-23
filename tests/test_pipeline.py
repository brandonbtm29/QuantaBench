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
    # sequential path: batched refinement is tested below with its own toy model
    r = pipeline.run_molecule(inp, cfg(tmp_path, method="gfn1-xtb", batched="off"), device="cuda:0", threads=1)
    assert r.status == "ok", r.error
    assert r.device == "cpu" and any("fell back to CPU" in n for n in r.notes)
    assert r.conformers[0].charges is not None and r.conformers[0].dipole_debye is not None


def test_use_batched_policy(tmp_path, monkeypatch):
    from quantabench.chem import batch_opt

    monkeypatch.setattr(batch_opt, "supports_batched", lambda m: m != "mmff")
    c = cfg(tmp_path)
    assert pipeline.use_batched(c, "gfn1-xtb", "cuda:0") and not pipeline.use_batched(c, "gfn1-xtb", "cpu")
    assert not pipeline.use_batched(c, "mmff", "cuda:0")
    assert pipeline.use_batched(cfg(tmp_path, batched="on"), "aimnet2", "cpu")
    assert not pipeline.use_batched(cfg(tmp_path, batched="off"), "aimnet2", "cuda:0")
    with pytest.raises(ValueError):
        pipeline.use_batched(cfg(tmp_path, batched="maybe"), "aimnet2", "cpu")


def test_batched_refine_on_gpu_with_cpu_fallback(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from quantabench.chem import batch_opt

    class GpuShy(batch_opt.HarmonicModel):
        def __init__(self, device, n):
            super().__init__(offset=0.05, charges=[-1.0 / n] * n)
            self.device = device

        def _compute(self, positions, index):
            if self.device != "cpu":
                raise RuntimeError("no GPU kernels for this device")
            return super()._compute(positions, index)

    calls = []
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    monkeypatch.setattr(engines, "supports", lambda key, numbers: (True, ""))
    monkeypatch.setattr(engines, "make_calculator", lambda *a, **k: pytest.fail("sequential path used"))
    monkeypatch.setattr(batch_opt, "make_batched_model",
                        lambda key, numbers, q, m, dev: calls.append(dev) or GpuShy(dev, len(numbers)))
    r = pipeline.run_molecule(from_smiles("OC(=O)C(F)(F)F", id="TFA"), cfg(tmp_path, method="gfn1-xtb"),
                              device="cuda:0", threads=1)
    assert r.status == "ok", r.error
    assert calls[0] == "cuda:0" and "cpu" in calls
    assert r.device == "cpu" and any("fell back to CPU" in n for n in r.notes)
    assert abs(sum(r.conformers[0].charges) + 1) < 1e-9 and r.conformers[0].dipole_debye is not None


def test_frequencies_mmff_and_spectrum_export(tmp_path):
    c = cfg(tmp_path, frequencies=True, formats=("xyz", "spectrum"))
    results, run_dir = pipeline.run([from_smiles("OC(=O)C(F)(F)F", id="TFA")], c)
    r = results[0]
    assert r.status == "ok", r.error
    vib = r.vibrations
    assert vib is not None and vib["n_modes"] == 3 * 7 - 6 and vib["method"] == "mmff"
    d = r.descriptors
    assert d["vib_n_imaginary"] == 0 and d["vib_zpe_kcal"] > 5 and d["vib_entropy_cal_molK"] > 50
    assert "frequencies" in r.timings
    folder = run_dir / "TFA"
    assert (folder / "TFA_ir.csv").exists() and (folder / "TFA_modes.csv").exists()
    import json

    summary = json.loads((folder / "TFA_summary.json").read_text())
    assert summary["vibrations"]["thermo"]["zpe_kcal"] == pytest.approx(d["vib_zpe_kcal"])
    assert "vib_gibbs_kcal" in (run_dir / "descriptors.csv").read_text()


def test_frequencies_leave_the_ranked_geometry_alone(tmp_path):
    """The tight pre-optimization for the Hessian moves atoms; the stored conformer (whose energy, charges and
    descriptors were computed earlier) must not be overwritten. The frequency geometry is exported apart."""
    from ase.io import read

    c = cfg(tmp_path, frequencies=True, formats=("xyz", "spectrum"), fmax=0.05)
    plain, _ = pipeline.run([from_smiles("OC(=O)C(F)(F)C(F)(F)F", id="PFPrA")], cfg(tmp_path, fmax=0.05,
                                                                                  formats=("xyz",)))
    results, run_dir = pipeline.run([from_smiles("OC(=O)C(F)(F)C(F)(F)F", id="PFPrA")], c)
    r, p0 = results[0], plain[0]
    assert r.status == "ok", r.error
    best, best0 = r.conformers[0], p0.conformers[0]
    pos = r.mol.GetConformer(best.conf_id).GetPositions()
    assert np.allclose(pos, p0.mol.GetConformer(best0.conf_id).GetPositions())
    assert best.energy_ev == pytest.approx(best0.energy_ev)
    freq_pos = np.array(r.vibrations["positions_A"])
    assert freq_pos.shape == pos.shape
    assert r.descriptors["vib_geometry_max_shift_A"] == pytest.approx(
        float(np.sqrt(((freq_pos - pos) ** 2).sum(1)).max()), abs=1e-5)
    folder = run_dir / "PFPrA"
    assert np.allclose(read(folder / "PFPrA.xyz").positions, pos, atol=1e-5)
    assert np.allclose(read(folder / "PFPrA_freq.xyz").positions, freq_pos, atol=1e-5)


def test_spectrum_format_without_frequencies_writes_nothing(tmp_path):
    results, run_dir = pipeline.run([from_smiles("CCO", id="eth")], cfg(tmp_path, formats=("xyz", "spectrum")))
    assert results[0].status == "ok"
    assert not list((run_dir / "eth").glob("*_ir.csv"))


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
