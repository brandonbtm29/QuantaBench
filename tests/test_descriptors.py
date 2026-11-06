"""Areas and volumes of the union of atom spheres (chem.descriptors, chem.radii): exact values for one and two
spheres, the radii sets and overrides, and the SASA regression (FreeSASA's ProtOr radii are 0 for non-protein
atoms, which made sasa_A2 the union of bare probe spheres)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA

from quantabench.chem import descriptors as d
from quantabench.chem import radii as R


def _mol(smiles: str, seed: int = 7) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def test_sphere_union_volume_matches_one_and_two_spheres():
    r = 1.7
    one = d.sphere_union_volume([[0, 0, 0]], [r])
    assert one == pytest.approx(4 / 3 * math.pi * r ** 3, rel=6e-3)          # grid: a lone sphere is the worst case
    dist = 1.5                                          # two equal spheres overlapping: two caps of height h
    h = r - dist / 2
    lens = 2 * math.pi * h ** 2 * (3 * r - h) / 3
    two = d.sphere_union_volume([[0, 0, 0], [dist, 0, 0]], [r, r])
    assert two == pytest.approx(2 * 4 / 3 * math.pi * r ** 3 - lens, rel=6e-3)
    assert d.sphere_union_volume([], []) == 0.0


def test_radii_sets_and_overrides():
    assert R.radii([1, 6, 7, 8, 9], "bondi") == pytest.approx([1.20, 1.70, 1.55, 1.52, 1.47])
    assert R.radii([1, 6, 7, 8, 9], "spartan") == pytest.approx([1.20, 1.92, 1.55, 1.52, 1.5138])
    assert R.radii([6, 9, 17], "bondi", {"C": 1.8, 9: 1.4, "cl": 1.9, "Xx": 3.0, "N": -1}) == pytest.approx([1.8, 1.4, 1.9])
    assert len(R.table("spartan")) == 111 and all(v > 0 for v in R.table("spartan").values())
    with pytest.raises(ValueError):
        R.radii([6], "uff")


def test_sasa_uses_element_radii_not_protor_zeros():
    m = _mol("OC(=O)C(F)(F)C(F)(F)C(F)(F)F")
    cid = m.GetConformer().GetId()
    assert set(rdFreeSASA.classifyAtoms(m)) == {0.0}            # what the old code passed
    probe_only = rdFreeSASA.CalcSASA(m, rdFreeSASA.classifyAtoms(m), confIdx=cid)
    out = d.surface_descriptors_3d(m, cid)
    assert out["sasa_A2"] > 1.6 * probe_only                      # 1.4 Å probe on Bondi radii, not on points
    assert out["vdw_area_A2"] < out["sasa_A2"]
    assert out["cpk_volume_A3"] > out["vdw_volume_A3"]            # Spartan's carbon is 1.92 Å
    # a single atom: the areas are spheres of radius r (vdW) and r + probe (SASA)
    ne = Chem.AddHs(Chem.MolFromSmiles("[Ne]"))
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    ne.AddConformer(conf, assignId=True)
    one = d.surface_descriptors_3d(ne, 0)
    r = R.default_radius(10, "bondi")
    assert one["vdw_area_A2"] == pytest.approx(4 * math.pi * r ** 2, rel=0.02)
    assert one["sasa_A2"] == pytest.approx(4 * math.pi * (r + 1.4) ** 2, rel=0.02)
    assert one["cpk_area_A2"] == pytest.approx(4 * math.pi * R.SPARTAN[10] ** 2, rel=0.02)


def test_surface_options_reach_the_descriptors():
    m = _mol("CCCCO")
    cid = m.GetConformer().GetId()
    base = d.conformer_descriptors(m, cid)
    spartan = d.conformer_descriptors(m, cid, d.SurfaceOptions.make("spartan"))
    small_probe = d.conformer_descriptors(m, cid, d.SurfaceOptions.make("bondi", {}, 1.0))
    assert spartan["vdw_volume_A3"] == pytest.approx(spartan["cpk_volume_A3"])
    assert spartan["vdw_volume_A3"] > base["vdw_volume_A3"]
    assert small_probe["sasa_A2"] < base["sasa_A2"] and small_probe["vdw_area_A2"] == pytest.approx(base["vdw_area_A2"])
    assert base["cpk_volume_A3"] == pytest.approx(spartan["cpk_volume_A3"])    # cpk_* never follow the options
    ens = d.ensemble_descriptors(m, [cid], np.array([1.0]), d.SurfaceOptions.make("spartan"))
    assert ens["vdw_volume_A3_best"] == pytest.approx(spartan["vdw_volume_A3"])
    assert {"sasa_A2_boltz", "cpk_area_A2_best", "cpk_volume_A3_boltz", "volume_A3_best"} <= set(ens)
