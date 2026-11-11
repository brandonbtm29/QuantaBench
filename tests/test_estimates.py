"""The cheap estimates of chem.estimates (DESC-2): Girolami's worked examples, the additive polarizabilities against
experimental values, the Sugden parachor, the packing density, coverage notes, and the pipeline columns."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench.chem import estimates as est
from quantabench.chem import pipeline
from quantabench.chem.inputs import from_smiles


def _mol(smiles: str, embed: bool = True) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if embed:
        AllChem.EmbedMolecule(m, randomSeed=7)
        AllChem.MMFFOptimizeMolecule(m)
    return m


# Girolami, J. Chem. Educ. 1994, 71, 962: his worked examples (calculated values, g/cm³)
@pytest.mark.parametrize("smiles, expected", [
    ("OC1CCCCC1", 0.92),            # cyclohexanol: one ring + one OH, factor 1.2
    ("CCP(C)C", 0.78),              # dimethylethylphosphine: no correction
    ("NCCN", 0.90),                 # ethylenediamine: two primary amines
    ("Brc1cccc2ccccc12", 1.49),     # 1-bromonaphthalene: two fused rings, 2 x 7.5 %
    ("O=S1(=O)CCCC1", 1.30),        # sulfolane: one ring + a sulfone
])
def test_girolami_reproduces_the_papers_examples(smiles, expected):
    rho, note = est.girolami_density(Chem.MolFromSmiles(smiles))
    assert note == "" and rho == pytest.approx(expected, abs=0.006)


def test_girolami_factor_is_capped_and_ions_are_left_out():
    f, why = est.girolami_factor(Chem.AddHs(Chem.MolFromSmiles("OCC(O)C(O)C(O)CO")))     # five OH: +50 %, capped
    assert f == pytest.approx(est.GIROLAMI_MAX) and why == ["5 hydroxyl"]
    f, why = est.girolami_factor(Chem.AddHs(Chem.MolFromSmiles("CC(=O)O")))              # the acid, not also an OH
    assert f == pytest.approx(1.1) and why == ["1 carboxylic acid"]
    f, _ = est.girolami_factor(Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC")))               # an amide, not an amine
    assert f == pytest.approx(1.1)
    assert est.girolami_density(Chem.MolFromSmiles("CC(=O)[O-]")) == (None, "Girolami density: not defined for an ion")
    assert est.girolami_density(Chem.MolFromSmiles("C[N+](=O)[O-]"))[0] > 1.0           # nitro: neutral overall
    rho, note = est.girolami_density(Chem.MolFromSmiles("[Xe]"))
    assert rho is None and "Xe" in note


# Experimental static polarizabilities (Å³), NIST CCCBDB (Computational Chemistry Comparison and Benchmark
# Database, release 22, "experimental polarizabilities"), read 2026-11-11
CCCBDB_ALPHA = {
    "C": 2.448, "CC": 4.226, "CCC": 5.921, "CCCC": 8.020, "c1ccccc1": 9.959, "O": 1.501, "CO": 3.210, "CCO": 5.112,
    "CC(C)=O": 6.270, "CC#N": 4.280, "ClC(Cl)(Cl)Cl": 10.002, "ClC(Cl)Cl": 8.129, "ClCCl": 6.659, "S=C=S": 8.749,
    "COC": 5.160, "C=O": 2.770, "CC(=O)O": 5.149, "N": 2.103, "CN": 3.754, "FC(F)(F)F": 2.824,
    "FC(F)(F)C(F)(F)F": 4.668, "FC(F)F": 2.801, "Fc1ccccc1": 10.200, "Fc1c(F)c(F)c(F)c(F)c1F": 9.580,
    "c1ccncc1": 9.493, "C[N+](=O)[O-]": 4.800, "CS(C)=O": 7.969,
}


@pytest.mark.parametrize("fn", [est.polarizability_bosque, est.polarizability_crippen])
def test_additive_polarizabilities_against_experiment(fn):
    errs = []
    for smiles, ref in CCCBDB_ALPHA.items():
        alpha, note = fn(Chem.MolFromSmiles(smiles))
        assert note == "", (smiles, note)
        errs.append((alpha - ref) / ref)
    errs = np.abs(errs)
    assert errs.mean() < 0.06 and errs.max() < 0.20       # Bosque-Sales 4.8 % (water 18 %), Crippen 3.8 % (methane 9 %)


def test_additive_polarizabilities_say_what_they_do_not_cover():
    for fn in (est.polarizability_bosque, est.polarizability_crippen):
        alpha, note = fn(Chem.MolFromSmiles("C[Si](C)(C)O[Si](C)(C)C"))
        assert alpha is None and "Si" in note
    assert est.ALPHA_A3_PER_MR == pytest.approx(0.396425, rel=1e-5)   # 3 / (4π N_A) in Å³ per cm³/mol


def test_sugden_parachor():
    # Sugden's values reproduce the measured parachors of simple liquids within a few units (Quayle 1953)
    assert est.parachor_sugden(Chem.MolFromSmiles("CCCCCC"))[0] == pytest.approx(268.2, abs=0.05)
    assert est.parachor_sugden(Chem.MolFromSmiles("c1ccccc1"))[0] == pytest.approx(207.1, abs=0.05)
    # explicit or implicit hydrogens give the same sum; an ester counts its -C(=O)O- as one group
    assert est.parachor_sugden(_mol("CC(=O)OC", embed=False))[0] == pytest.approx(
        est.parachor_sugden(Chem.MolFromSmiles("CC(=O)OC"))[0])
    assert est.parachor_sugden(Chem.MolFromSmiles("CC(=O)OC"))[0] == pytest.approx(3 * 4.8 + 6 * 17.1 + 60.0)
    p, note = est.parachor_sugden(Chem.MolFromSmiles("C[Sn](C)(C)C"))
    assert p is None and "Sn" in note


def test_packing_density_of_liquids_it_was_not_fitted_to_check():
    # measured at 25 °C (CRC Handbook): hexane 0.655, perfluorohexane 1.669, ethanol 0.785, chlorobenzene 1.101
    for smiles, rho in (("CCCCCC", 0.655), ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", 1.669),
                        ("CCO", 0.785), ("Clc1ccccc1", 1.101)):
        d, notes = est.estimate(_mol(smiles), d4=False)
        assert d["density_packing_g_cm3"] == pytest.approx(rho, rel=0.06), smiles
        assert d["molar_volume_packing_cm3"] * d["density_packing_g_cm3"] == pytest.approx(
            Chem.Descriptors.MolWt(_mol(smiles)), rel=1e-9)
        assert not notes
    rho, note = est.packing_density(_mol("CC(=O)[O-]"), 60.0)
    assert rho is None and "ion" in note
    rho, note = est.packing_density(_mol("C[Se]C"), 60.0)
    assert rho is None and "Se" in note


def test_estimate_collects_values_and_notes():
    d, notes = est.estimate(_mol("C[Si](C)(C)C"), d4=False)
    assert {"density_girolami_g_cm3", "molar_volume_girolami_cm3", "parachor_sugden"} <= set(d)
    assert "polarizability_bosque_A3" not in d and "polarizability_crippen_A3" not in d
    assert any("Bosque-Sales" in n for n in notes) and any("Crippen" in n for n in notes)
    d, notes = est.estimate(_mol("CCO", embed=False), d4=False)      # no conformer: the graph-only estimates
    assert "density_packing_g_cm3" not in d and "density_girolami_g_cm3" in d and not notes


def test_d4_polarizability():
    pytest.importorskip("torch")
    if not est.d4_available():
        pytest.skip("neither tad-dftd4 nor the dxtb fork")
    m = _mol("c1ccccc1")
    alpha, note = est.polarizability_d4([a.GetAtomicNum() for a in m.GetAtoms()], m.GetConformer().GetPositions())
    assert note == "" and alpha == pytest.approx(CCCBDB_ALPHA["c1ccccc1"], rel=0.10)


def test_pipeline_writes_the_estimates_and_both_xtb_polarizabilities(tmp_path):
    cfg = pipeline.BuildConfig(method="mmff", rank_method="refine", max_conformers=3, refine_top=2,
                               xtb_properties=True, output_dir=str(tmp_path / "runs"), write_files=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(est, "d4_available", lambda: False)         # D4 is optional and has its own test
        r = pipeline.run_molecule(from_smiles("CCCCO", id="butanol"), cfg, device="cpu", threads=1)
    assert r.status == "ok", r.error
    d = r.descriptors
    assert 0.7 < d["density_girolami_g_cm3"] < 0.9 and 0.7 < d["density_packing_g_cm3"] < 0.9
    assert 8.0 < d["polarizability_bosque_A3"] < 9.5 and 8.0 < d["polarizability_crippen_A3"] < 9.5
    assert d["parachor_sugden"] > 200
    from quantabench.chem import properties

    if properties.tblite_available():       # GFN2-xTB's response although MMFF refined it, and GFN1-xTB's
        assert 0 < d["tblite_gfn2_polarizability_iso_A3"] < d["polarizability_bosque_A3"]
        if not properties.available():      # no dxtb: GFN1-xTB from tblite
            assert d["xtb_polarizability_method"].startswith("tblite GFN1-xTB finite field")
        assert d["xtb_polarizability_iso_A3"] > 0


def test_pipeline_notes_an_estimate_without_parameters(tmp_path):
    cfg = pipeline.BuildConfig(method="mmff", rank_method="refine", max_conformers=2, refine_top=1,
                               xtb_properties=False, output_dir=str(tmp_path / "runs"), write_files=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(est, "d4_available", lambda: False)
        r = pipeline.run_molecule(from_smiles("C[Si](C)(C)C", id="TMS"), cfg, device="cpu", threads=1)
    assert r.status == "ok", r.error
    assert "polarizability_bosque_A3" not in r.descriptors and "density_girolami_g_cm3" in r.descriptors
    assert any(n.startswith("Bosque-Sales polarizability: no parameter for Si") for n in r.notes)
